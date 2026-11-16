"""
grid.py — cached occupancy-grid representation.

One GridMap is built per received OccupancyGrid message. The NumPy array and
every derived mask (free / unknown / occupied / inflated obstacles) is
computed at most once per map and reused by all candidate evaluations.

Conventions (standard nav_msgs/OccupancyGrid):
  value  < 0                       unknown
  0 <= value <= free_max           free
  value >= occupied_min            occupied
  anything in between              uncertain (not free, not an obstacle)

The grid origin is assumed to be axis-aligned (identity orientation), which
is what slam_toolbox, Cartographer, gmapping and Nav2 costmaps publish.
"""

import math

import numpy as np

try:                                    # SciPy is optional but recommended
    from scipy import ndimage as _ndimage
except Exception:                       # pragma: no cover - depends on host
    _ndimage = None


def _dilate_octagon(mask, radius_cells):
    """Grow a mask by about radius_cells using NumPy only.

    Fallback for obstacle inflation when SciPy is unavailable. Alternating
    4- and 8-neighbor steps approximate a disk by an octagon.

    Args:
        mask: Boolean mask to grow.
        radius_cells: Growth radius [cells].

    Returns:
        The dilated boolean mask.
    """
    out = mask.copy()
    steps = int(math.ceil(radius_cells))
    for i in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        if i % 2 == 1:                  # alternate 4-/8-neighborhood
            grown[1:, 1:] |= out[:-1, :-1]
            grown[1:, :-1] |= out[:-1, 1:]
            grown[:-1, 1:] |= out[1:, :-1]
            grown[:-1, :-1] |= out[1:, 1:]
        out = grown
    return out


class GridMap:
    """Snapshot of one occupancy grid with lazily cached masks.

    Cell (row, col) covers x in [origin_x + col*res, origin_x + (col+1)*res)
    and y likewise with row; cell centers are returned by cell_to_world().
    The masks are computed on first access and then reused.

    Attributes:
        grid: Occupancy values as an int8 array of shape (height, width).
        width: Number of columns.
        height: Number of rows.
        resolution: Cell size [m].
        origin_x: x of the grid's lower-left corner in the map frame [m].
        origin_y: y of the grid's lower-left corner in the map frame [m].
        frame_id: Frame of the grid (normally 'map').
        seq: Sequence number of the map, assigned by the caller.
        free_max: Highest value still considered free.
        occupied_min: Lowest value considered occupied.
    """

    def __init__(self, data, width, height, resolution,
                 origin_x, origin_y, frame_id='map', seq=0,
                 free_max=25, occupied_min=50):
        """Copy an occupancy grid into a new GridMap.

        Args:
            data: Flat sequence of cell values in row-major order, as in
                nav_msgs/OccupancyGrid.data.
            width: Number of columns.
            height: Number of rows.
            resolution: Cell size [m].
            origin_x: x of the grid origin in the map frame [m].
            origin_y: y of the grid origin in the map frame [m].
            frame_id: Frame of the grid; empty falls back to 'map'.
            seq: Sequence number of the map.
            free_max: Highest value considered free.
            occupied_min: Lowest value considered occupied.

        Raises:
            ValueError: If the size or resolution is not positive.
        """
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame_id = str(frame_id) if frame_id else 'map'
        self.seq = int(seq)
        self.free_max = int(free_max)
        self.occupied_min = int(occupied_min)

        if self.width <= 0 or self.height <= 0 or self.resolution <= 0.0:
            raise ValueError('invalid occupancy-grid metadata')

        # Own copy: independent of the ROS message buffer.
        self.grid = np.array(data, dtype=np.int8).reshape(
            self.height, self.width)

        self._free = None
        self._unknown = None
        self._occupied = None
        self._known_cells = None
        self._obstacle_distance = None
        self._mask_cache = {}

    # ── basic masks (lazy, cached) ───────────────────────────────────────
    @property
    def free(self):
        """Boolean mask of free cells (0 <= value <= free_max)."""
        if self._free is None:
            self._free = (self.grid >= 0) & (self.grid <= self.free_max)
        return self._free

    @property
    def unknown(self):
        """Boolean mask of unknown cells (value < 0)."""
        if self._unknown is None:
            self._unknown = self.grid < 0
        return self._unknown

    @property
    def occupied(self):
        """Boolean mask of occupied cells (value >= occupied_min)."""
        if self._occupied is None:
            self._occupied = self.grid >= self.occupied_min
        return self._occupied

    @property
    def known_cells(self):
        """Number of known cells (free, uncertain or occupied)."""
        if self._known_cells is None:
            self._known_cells = int(np.count_nonzero(self.grid >= 0))
        return self._known_cells

    @property
    def known_area_m2(self):
        """Known area of the map [m^2]."""
        return float(self.known_cells) * self.resolution * self.resolution

    # ── geometry ─────────────────────────────────────────────────────────
    def world_to_cell(self, x, y):
        """Return the (row, col) of the cell containing map point (x, y).

        The result may lie outside the grid; check it with in_bounds().
        """
        col = int(math.floor((float(x) - self.origin_x) / self.resolution))
        row = int(math.floor((float(y) - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, row, col):
        """Return the map-frame (x, y) of a cell center [m]."""
        return (
            self.origin_x + (float(col) + 0.5) * self.resolution,
            self.origin_y + (float(row) + 0.5) * self.resolution,
        )

    def cells_to_world(self, rows, cols):
        """Vectorized cell_to_world(): return arrays (xs, ys) [m]."""
        xs = self.origin_x + (np.asarray(cols, dtype=float) + 0.5) * self.resolution
        ys = self.origin_y + (np.asarray(rows, dtype=float) + 0.5) * self.resolution
        return xs, ys

    def in_bounds(self, row, col):
        """True if (row, col) lies inside the grid."""
        return 0 <= row < self.height and 0 <= col < self.width

    # ── obstacle inflation ───────────────────────────────────────────────
    def obstacle_mask(self, radius_m):
        """Return the mask of cells within radius_m of a known obstacle.

        Uses an exact Euclidean distance transform when SciPy is available
        and an octagonal approximation otherwise. Results are cached per
        radius, so all candidates of one map share the same computation.

        Args:
            radius_m: Inflation radius [m]; 0 returns the occupied mask.

        Returns:
            Boolean mask; True marks a cell too close to an obstacle.
        """
        radius_cells = max(0.0, float(radius_m) / self.resolution)
        key = round(radius_cells, 3)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        occupied = self.occupied
        if radius_cells <= 0.0 or not occupied.any():
            mask = occupied
        elif _ndimage is not None:
            if self._obstacle_distance is None:
                self._obstacle_distance = _ndimage.distance_transform_edt(
                    ~occupied)
            mask = self._obstacle_distance <= radius_cells
        else:
            mask = _dilate_octagon(occupied, radius_cells)

        self._mask_cache[key] = mask
        return mask

    def is_clear(self, x, y, radius_m):
        """Check that no known obstacle lies within radius_m of (x, y).

        Args:
            x: Point x in the map frame [m].
            y: Point y in the map frame [m].
            radius_m: Required clearance [m].

        Returns:
            True if the point is inside the grid and clear; False otherwise.
        """
        row, col = self.world_to_cell(x, y)
        if not self.in_bounds(row, col):
            return False
        return not bool(self.obstacle_mask(radius_m)[row, col])

    # ── local statistics ─────────────────────────────────────────────────
    def count_known_in_window(self, x, y, radius_m):
        """Count known cells in the square window around (x, y).

        Args:
            x: Window center x in the map frame [m].
            y: Window center y in the map frame [m].
            radius_m: Half the window side [m].

        Returns:
            Number of known cells in the window (clipped to the grid).
        """
        row, col = self.world_to_cell(x, y)
        r = int(math.ceil(float(radius_m) / self.resolution))
        r0 = max(0, row - r)
        r1 = min(self.height, row + r + 1)
        c0 = max(0, col - r)
        c1 = min(self.width, col + r + 1)
        if r0 >= r1 or c0 >= c1:
            return 0
        return int(np.count_nonzero(self.grid[r0:r1, c0:c1] >= 0))

    # ── alignment helpers for map-to-map comparison ──────────────────────
    @property
    def origin_cell_x(self):
        """Grid origin x in global cell units (for map-to-map alignment)."""
        return int(round(self.origin_x / self.resolution))

    @property
    def origin_cell_y(self):
        """Grid origin y in global cell units (for map-to-map alignment)."""
        return int(round(self.origin_y / self.resolution))


def map_change_pct(previous, current):
    """Percentage of known cells that differ between two maps.

    New known cells count as change; pure growth by UNKNOWN cells does not.
    The comparison is aligned in global cell coordinates, so a moving map
    origin or a resized grid does not create a false 100 % change.

    Args:
        previous: Earlier GridMap, or None.
        current: Later GridMap, or None.

    Returns:
        Changed share of the union of known cells [%]; 100 if the
        resolutions differ, None when either map is missing.
    """
    if previous is None or current is None:
        return None
    if abs(previous.resolution - current.resolution) > 1e-9:
        return 100.0

    p = previous.grid
    c = current.grid

    px0, py0 = previous.origin_cell_x, previous.origin_cell_y
    px1, py1 = px0 + previous.width, py0 + previous.height
    cx0, cy0 = current.origin_cell_x, current.origin_cell_y
    cx1, cy1 = cx0 + current.width, cy0 + current.height

    ix0, iy0 = max(px0, cx0), max(py0, cy0)
    ix1, iy1 = min(px1, cx1), min(py1, cy1)

    prev_known_total = previous.known_cells
    curr_known_total = current.known_cells

    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0 if (prev_known_total + curr_known_total) == 0 else 100.0

    p_sub = p[iy0 - py0:iy1 - py0, ix0 - px0:ix1 - px0]
    c_sub = c[iy0 - cy0:iy1 - cy0, ix0 - cx0:ix1 - cx0]

    p_known = p_sub >= 0
    c_known = c_sub >= 0
    union_overlap = p_known | c_known

    changed_overlap = int(np.count_nonzero((p_sub != c_sub) & union_overlap))
    # Known cells outside the overlap appeared or disappeared with a resize.
    prev_outside = max(0, prev_known_total - int(np.count_nonzero(p_known)))
    curr_outside = max(0, curr_known_total - int(np.count_nonzero(c_known)))

    changed = changed_overlap + prev_outside + curr_outside
    union_known = (
        int(np.count_nonzero(union_overlap)) + prev_outside + curr_outside)

    if union_known <= 0:
        return 0.0
    return 100.0 * float(changed) / float(union_known)
