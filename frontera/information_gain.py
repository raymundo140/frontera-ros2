"""
information_gain.py — expected newly observed area from a viewpoint.

Estimator: bounded 2-D ray casting on the occupancy grid.

From the candidate goal position, `num_rays` rays are cast over the sensor's
horizontal field of view up to `sensor_range_m`. A ray stops at the first
known OCCUPIED cell. Every UNKNOWN sample it crosses before that contributes
its polar area element (r * dtheta * dr). The sum is the area [m^2] of
unknown space that is in range and not occluded by a known obstacle.

Assumptions (documented, not hidden):
* Unknown space is treated as transparent, so this is an optimistic upper
  bound: the real gain is smaller if a wall hides inside the unknown area.
* Cells outside the current grid are unknown (the map can still grow).
* The sensor is modeled as a planar range sensor at the goal position with a
  given range and field of view. For a field of view below 360 degrees the
  viewing direction must be supplied by the caller.
* Uncertain cells (between the free and occupied thresholds) neither block a
  ray nor count as unknown.
"""

import math

import numpy as np


def reference_gain_m2(sensor_range_m, fov_rad):
    """Area of the full sensor footprint; used to normalize gain to [0, 1]."""
    return 0.5 * float(fov_rad) * float(sensor_range_m) ** 2


def estimate_gain_m2(grid, x, y, sensor_range_m, fov_rad=2.0 * math.pi,
                     view_bearing=0.0, num_rays=72, step_m=0.0):
    """Return the visible unknown area [m^2] from (x, y)."""
    res = grid.resolution
    step = float(step_m) if step_m and step_m > 0.0 else res
    step = max(step, res * 0.5)

    fov = min(float(fov_rad), 2.0 * math.pi)
    n = int(num_rays)
    dtheta = fov / n
    angles = (
        float(view_bearing) - 0.5 * fov + (np.arange(n) + 0.5) * dtheta)

    radii = np.arange(step, float(sensor_range_m) + 1e-9, step)
    if radii.size == 0:
        return 0.0

    xs = float(x) + np.outer(np.cos(angles), radii)
    ys = float(y) + np.outer(np.sin(angles), radii)

    cols = np.floor((xs - grid.origin_x) / res).astype(np.int64)
    rows = np.floor((ys - grid.origin_y) / res).astype(np.int64)

    inside = (
        (rows >= 0) & (rows < grid.height) &
        (cols >= 0) & (cols < grid.width)
    )

    values = np.full(rows.shape, -1, dtype=np.int16)    # outside = unknown
    values[inside] = grid.grid[rows[inside], cols[inside]]

    blocked = values >= grid.occupied_min
    occluded = np.maximum.accumulate(blocked, axis=1)

    visible_unknown = (values < 0) & ~occluded

    area_elements = radii * dtheta * step                # per sample on a ray
    return float((visible_unknown * area_elements[np.newaxis, :]).sum())
