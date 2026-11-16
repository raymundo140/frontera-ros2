"""
frontier_detection.py — frontier cells and frontier clusters.

Definition (the classical one of Yamauchi, 1997, restricted to 4-connectivity):
  a frontier cell is a FREE cell with at least one CARDINAL (4-connected)
  UNKNOWN neighbor. Diagonal-only adjacency is ignored so that unknown space
  touching free space across an obstacle corner does not create a frontier.

Cells are grouped into 8-connected clusters. Clusters shorter than a minimum
physical length are discarded as map noise, and very long clusters are split
into spatially compact pieces so that a wide opening yields more than one
candidate destination.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage as _ndimage
except Exception:                       # pragma: no cover - depends on host
    _ndimage = None

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass
class FrontierCluster:
    """One candidate frontier: a connected group of frontier cells.

    Cell indices refer to the occupancy grid (row = y, col = x). A long
    connected frontier may be split into several clusters; each piece keeps
    the size of the component it came from in parent_size.
    """

    rows: np.ndarray            # frontier cell rows   (int)
    cols: np.ndarray            # frontier cell cols   (int)
    centroid_row: float
    centroid_col: float
    size: int                   # number of frontier cells in this candidate
    length_m: float             # size * resolution (approximate length)
    parent_size: int            # cells of the connected cluster it came from


def detect_frontier_mask(grid):
    """Return a boolean mask of the frontier cells of a grid.

    A frontier cell is a free cell with at least one unknown 4-connected
    neighbor (see the module docstring).

    Args:
        grid: GridMap holding the current occupancy grid.

    Returns:
        Boolean array of the grid's shape; True marks a frontier cell.
    """
    free = grid.free
    unknown = grid.unknown

    has_unknown = np.zeros_like(unknown)
    has_unknown[1:, :] |= unknown[:-1, :]
    has_unknown[:-1, :] |= unknown[1:, :]
    has_unknown[:, 1:] |= unknown[:, :-1]
    has_unknown[:, :-1] |= unknown[:, 1:]

    return free & has_unknown


def _label_groups(mask):
    """Split a mask into its 8-connected components.

    Uses SciPy when available and a breadth-first traversal otherwise; both
    yield the same components.

    Args:
        mask: Boolean grid mask.

    Returns:
        List of (rows, cols) index arrays, one pair per component.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return []

    if _ndimage is not None:
        labels, count = _ndimage.label(mask, structure=_EIGHT_CONNECTED)
        if count == 0:
            return []
        lab = labels[rows, cols]
        order = np.argsort(lab, kind='stable')
        lab_sorted = lab[order]
        cuts = np.flatnonzero(np.diff(lab_sorted)) + 1
        return [
            (rows[idx], cols[idx])
            for idx in np.split(order, cuts)
        ]

    # Pure-Python fallback (breadth-first traversal).
    cells = list(zip(rows.tolist(), cols.tolist()))
    cell_set = set(cells)
    visited = set()
    groups = []
    for cell in cells:
        if cell in visited:
            continue
        queue = deque([cell])
        visited.add(cell)
        members = []
        while queue:
            r, c = queue.popleft()
            members.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = (r + dr, c + dc)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
        arr = np.array(members, dtype=np.int64)
        groups.append((arr[:, 0], arr[:, 1]))
    return groups


def _make_cluster(rows, cols, resolution, parent_size):
    """Build a FrontierCluster from its cell indices."""
    return FrontierCluster(
        rows=rows,
        cols=cols,
        centroid_row=float(rows.mean()),
        centroid_col=float(cols.mean()),
        size=int(rows.size),
        length_m=float(rows.size) * resolution,
        parent_size=int(parent_size),
    )


def cluster_frontiers(grid, frontier_mask, min_length_m, max_length_m):
    """Group frontier cells into candidate clusters.

    Connected components shorter than min_length_m are discarded as map
    noise. Components longer than max_length_m are cut into world-anchored
    square tiles, so a wide opening yields several candidates and the same
    physical frontier is split the same way on successive maps.

    Args:
        grid: GridMap holding the current occupancy grid.
        frontier_mask: Boolean mask from detect_frontier_mask().
        min_length_m: Minimum frontier length [m] (cells x resolution).
        max_length_m: Maximum frontier length before splitting [m];
            0 or less disables splitting.

    Returns:
        Tuple (clusters, rejected_small): the list of FrontierCluster
        candidates and the number of components discarded as too short.
    """
    res = grid.resolution
    min_cells = max(1, int(math.ceil(float(min_length_m) / res)))

    clusters = []
    rejected_small = 0

    for rows, cols in _label_groups(frontier_mask):
        n = int(rows.size)
        if n < min_cells:
            rejected_small += 1
            continue

        if max_length_m <= 0.0 or n * res <= max_length_m:
            clusters.append(_make_cluster(rows, cols, res, n))
            continue

        # Split a long frontier with world-anchored square tiles of side
        # max_length_m. Tiles are anchored to world coordinates so the same
        # physical frontier produces the same pieces on successive maps.
        xs, ys = grid.cells_to_world(rows, cols)
        tx = np.floor(xs / max_length_m).astype(np.int64)
        ty = np.floor(ys / max_length_m).astype(np.int64)
        # Combine both tile indices into one sortable key (large prime stride).
        key = tx * 1000003 + ty
        order = np.argsort(key, kind='stable')
        key_sorted = key[order]
        cuts = np.flatnonzero(np.diff(key_sorted)) + 1
        pieces = np.split(order, cuts)

        kept = [p for p in pieces if p.size >= min_cells]
        if not kept:
            # Every fragment is tiny: keep the whole cluster as one candidate.
            clusters.append(_make_cluster(rows, cols, res, n))
            continue
        for piece in kept:
            clusters.append(_make_cluster(rows[piece], cols[piece], res, n))

    return clusters, rejected_small
