"""
reachability.py — obstacle-aware path distance from the robot.

ONE shortest-path computation (Dijkstra wavefront) from the robot's cell
gives the approximate path length to every frontier candidate at once, and
tells which candidates are not connected to the robot through known free
space. It complements the navigation stack's planner; it does not replace it:

* It is an estimate on the SLAM occupancy grid only (no dynamic obstacles).
* Obstacles are inflated by a radius slightly BELOW the robot's inscribed
  radius, and the grid is coarsened optimistically (a coarse cell is
  traversable if ANY of its fine cells is), so narrow but passable gaps stay
  open. A false "reachable" costs one failed navigation goal, which the
  exploration memory absorbs; a false "unreachable" would lose map area.
* A candidate the wavefront cannot reach is never discarded permanently on
  this evidence alone — see the probing logic in controller.py.
"""

import heapq
import math

import numpy as np

try:
    from scipy.sparse import csr_matrix as _csr_matrix
    from scipy.sparse.csgraph import dijkstra as _dijkstra
except Exception:                       # pragma: no cover - depends on host
    _csr_matrix = None
    _dijkstra = None

_NO_PRED = -9999
_SQRT2 = math.sqrt(2.0)


class ReachabilityField:
    """Distance field [m] over a coarsened traversability grid.

    Built by compute_reachability(). Queries take fine-grid (row, col)
    indices of the original occupancy grid and map them to coarse cells.

    Attributes:
        grid: The GridMap the field was computed on.
        factor: Fine cells per coarse cell along each axis.
        coarse_shape: (rows, cols) of the coarse grid.
        cell_m: Coarse cell size [m].
        dist: Flat array of path lengths from the start cell [m]; inf
            where unreachable.
        pred: Flat array of predecessor indices along the shortest paths;
            negative at the start and at unreachable cells.
        start_index: Flat index of the start cell.
        start_offset_m: Distance from the robot to the start cell [m],
            added to every reported path length.
    """

    def __init__(self, grid, factor, coarse_shape, dist, pred,
                 start_index, start_offset_m):
        """Store a computed field; see compute_reachability()."""
        self.grid = grid
        self.factor = int(factor)
        self.coarse_shape = coarse_shape
        self.cell_m = grid.resolution * self.factor
        self.dist = dist                # flat array, inf where unreachable
        self.pred = pred                # flat predecessor indices
        self.start_index = int(start_index)
        self.start_offset_m = float(start_offset_m)

    def _index(self, row, col):
        """Return the flat coarse index of a fine cell, or None if outside."""
        cr = int(row) // self.factor
        cc = int(col) // self.factor
        if not (0 <= cr < self.coarse_shape[0]
                and 0 <= cc < self.coarse_shape[1]):
            return None
        return cr * self.coarse_shape[1] + cc

    def distances(self, rows, cols):
        """Return path lengths to many fine-grid cells at once.

        Args:
            rows: Array of fine-grid row indices.
            cols: Array of fine-grid column indices.

        Returns:
            Array of path lengths from the robot [m]; inf where a cell is
            unreachable or outside the grid.
        """
        cr = np.asarray(rows) // self.factor
        cc = np.asarray(cols) // self.factor
        ok = (
            (cr >= 0) & (cr < self.coarse_shape[0]) &
            (cc >= 0) & (cc < self.coarse_shape[1])
        )
        out = np.full(cr.shape, np.inf)
        idx = cr[ok] * self.coarse_shape[1] + cc[ok]
        out[ok] = self.dist[idx] + self.start_offset_m
        return out

    def distance(self, row, col):
        """Return the path length from the robot to one fine cell [m].

        Returns:
            Path length [m]; math.inf if unreachable or outside the grid.
        """
        idx = self._index(row, col)
        if idx is None:
            return math.inf
        d = float(self.dist[idx])
        return d + self.start_offset_m if math.isfinite(d) else math.inf

    def _coarse_world(self, idx):
        """Return the map-frame (x, y) of a coarse cell center [m]."""
        cr, cc = divmod(int(idx), self.coarse_shape[1])
        return (
            self.grid.origin_x + (cc + 0.5) * self.cell_m,
            self.grid.origin_y + (cr + 0.5) * self.cell_m,
        )

    def path_direction(self, row, col, robot_x, robot_y, lookahead_m):
        """Return the initial direction of the path to a cell.

        Follows the shortest path back from the target to the point about
        lookahead_m from the start, and returns the direction from the robot
        to that point. This is the direction the robot would initially
        travel, which for an omnidirectional base is independent of its
        chassis heading; it feeds the directional continuity factor C.

        Args:
            row: Target fine-grid row.
            col: Target fine-grid column.
            robot_x: Robot x in the map frame [m].
            robot_y: Robot y in the map frame [m].
            lookahead_m: Path length that defines the initial direction [m].

        Returns:
            Unit vector (x, y) in the map frame, or None when no path
            exists or the direction is undefined.
        """
        idx = self._index(row, col)
        if idx is None or not math.isfinite(float(self.dist[idx])):
            return None

        # Walk back along predecessors; the guard bounds the loop even if
        # the predecessor array were corrupted.
        node = idx
        guard = self.dist.size + 1
        while guard > 0:
            guard -= 1
            if float(self.dist[node]) <= lookahead_m:
                break
            prev = int(self.pred[node])
            if prev < 0:
                break
            node = prev

        wx, wy = self._coarse_world(node)
        dx = wx - float(robot_x)
        dy = wy - float(robot_y)
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None
        return dx / norm, dy / norm


def _coarsen_any(mask, factor):
    """Coarsen a mask by factor; a coarse cell is True if ANY fine cell is.

    The grid is padded with False to a multiple of factor first.
    """
    if factor <= 1:
        return mask
    h, w = mask.shape
    hp = int(math.ceil(h / factor)) * factor
    wp = int(math.ceil(w / factor)) * factor
    if hp != h or wp != w:
        padded = np.zeros((hp, wp), dtype=bool)
        padded[:h, :w] = mask
        mask = padded
    return mask.reshape(hp // factor, factor, wp // factor, factor).any(
        axis=(1, 3))


def _nearest_traversable(coarse, row, col, radius_cells):
    """Find the traversable coarse cell nearest to (row, col).

    Needed because the robot's own cell may lie inside an inflated obstacle
    (for example right next to a wall).

    Args:
        coarse: Coarse traversability mask.
        row: Coarse row of the robot.
        col: Coarse column of the robot.
        radius_cells: Search radius [coarse cells].

    Returns:
        Tuple (row, col, distance in cells), or None if no traversable
        cell lies within the radius.
    """
    h, w = coarse.shape
    r0 = max(0, row - radius_cells)
    r1 = min(h, row + radius_cells + 1)
    c0 = max(0, col - radius_cells)
    c1 = min(w, col + radius_cells + 1)
    if r0 >= r1 or c0 >= c1:
        return None
    rr, cc = np.nonzero(coarse[r0:r1, c0:c1])
    if rr.size == 0:
        return None
    rr = rr + r0
    cc = cc + c0
    d2 = (rr - row) ** 2 + (cc - col) ** 2
    best = int(np.argmin(d2))
    if d2[best] > radius_cells * radius_cells:
        return None
    return int(rr[best]), int(cc[best]), math.sqrt(float(d2[best]))


def _dijkstra_scipy(coarse, start_index, cell_m):
    """Run Dijkstra on the 8-connected coarse grid using SciPy.

    Returns:
        Tuple (dist, pred) of flat arrays; see ReachabilityField.
    """
    h, w = coarse.shape
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)

    src, dst, wgt = [], [], []

    horiz = coarse[:, :-1] & coarse[:, 1:]
    src.append(idx[:, :-1][horiz])
    dst.append(idx[:, 1:][horiz])
    wgt.append(np.full(int(horiz.sum()), cell_m))

    vert = coarse[:-1, :] & coarse[1:, :]
    src.append(idx[:-1, :][vert])
    dst.append(idx[1:, :][vert])
    wgt.append(np.full(int(vert.sum()), cell_m))

    # Diagonal moves only through a fully traversable 2x2 block
    # (no cutting across obstacle corners).
    block = (
        coarse[:-1, :-1] & coarse[1:, 1:] &
        coarse[:-1, 1:] & coarse[1:, :-1]
    )
    n_block = int(block.sum())
    src.append(idx[:-1, :-1][block])
    dst.append(idx[1:, 1:][block])
    wgt.append(np.full(n_block, cell_m * _SQRT2))
    src.append(idx[:-1, 1:][block])
    dst.append(idx[1:, :-1][block])
    wgt.append(np.full(n_block, cell_m * _SQRT2))

    graph = _csr_matrix(
        (np.concatenate(wgt), (np.concatenate(src), np.concatenate(dst))),
        shape=(h * w, h * w))

    dist, pred = _dijkstra(
        graph, directed=False, indices=start_index,
        return_predecessors=True)
    return dist, pred


def _dijkstra_python(coarse, start_index, cell_m):
    """Pure-Python Dijkstra, equivalent to _dijkstra_scipy()."""
    h, w = coarse.shape
    n = h * w
    dist = np.full(n, np.inf)
    pred = np.full(n, _NO_PRED, dtype=np.int64)
    flat = coarse.ravel()
    dist[start_index] = 0.0
    heap = [(0.0, start_index)]
    diag = cell_m * _SQRT2

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        r, c = divmod(node, w)
        for dr, dc, cost in (
                (-1, 0, cell_m), (1, 0, cell_m),
                (0, -1, cell_m), (0, 1, cell_m),
                (-1, -1, diag), (-1, 1, diag),
                (1, -1, diag), (1, 1, diag)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nb = nr * w + nc
            if not flat[nb]:
                continue
            if dr != 0 and dc != 0:
                # Diagonal only if both side cells are free (no corner cut).
                if not (flat[r * w + nc] and flat[nr * w + c]):
                    continue
            nd = d + cost
            if nd < dist[nb]:
                dist[nb] = nd
                pred[nb] = node
                heapq.heappush(heap, (nd, nb))
    return dist, pred


def compute_reachability(grid, robot_x, robot_y, inflation_m,
                         coarse_resolution_m, start_search_m,
                         force_python=False):
    """Compute path lengths from the robot to every reachable cell.

    Free cells farther than inflation_m from any known obstacle are
    traversable. The grid is coarsened optimistically, and a single
    Dijkstra wavefront from the robot's cell then gives the path length to
    all frontier candidates at once.

    Args:
        grid: GridMap holding the current occupancy grid.
        robot_x: Robot x in the map frame [m].
        robot_y: Robot y in the map frame [m].
        inflation_m: Obstacle inflation radius [m]; keep it slightly below
            the robot's inscribed radius.
        coarse_resolution_m: Target coarse cell size [m]; reduced to
            inflation_m when larger.
        start_search_m: How far from the robot to look for a traversable
            start cell [m].
        force_python: Use the pure-Python Dijkstra even if SciPy exists
            (for tests).

    Returns:
        A ReachabilityField, or None when no traversable cell lies within
        start_search_m of the robot.
    """
    res = grid.resolution

    traversable = grid.free & ~grid.obstacle_mask(inflation_m)

    # Coarse cell no larger than the inflation radius: an inflated wall is
    # then always thicker than a coarse cell and cannot be "jumped".
    target = float(coarse_resolution_m)
    if inflation_m > 0.0:
        target = min(target, float(inflation_m))
    factor = max(1, int(math.floor(target / res + 1e-9)))

    coarse = _coarsen_any(traversable, factor)
    cell_m = res * factor

    row, col = grid.world_to_cell(robot_x, robot_y)
    crow, ccol = row // factor, col // factor
    radius_cells = max(1, int(math.ceil(float(start_search_m) / cell_m)))

    start = _nearest_traversable(coarse, crow, ccol, radius_cells)
    if start is None:
        return None
    srow, scol, offset_cells = start
    start_index = srow * coarse.shape[1] + scol

    if _dijkstra is not None and not force_python:
        dist, pred = _dijkstra_scipy(coarse, start_index, cell_m)
    else:
        dist, pred = _dijkstra_python(coarse, start_index, cell_m)

    return ReachabilityField(
        grid=grid,
        factor=factor,
        coarse_shape=coarse.shape,
        dist=np.asarray(dist, dtype=float),
        pred=np.asarray(pred),
        start_index=start_index,
        start_offset_m=offset_cells * cell_m,
    )
