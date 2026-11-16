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
    """Return the area of the full sensor footprint.

    Used to normalize the information gain to [0, 1].

    Args:
        sensor_range_m: Sensor range [m].
        fov_rad: Horizontal field of view [rad].

    Returns:
        Footprint area 0.5 * fov * range^2 [m^2].
    """
    return 0.5 * float(fov_rad) * float(sensor_range_m) ** 2


def estimate_gain_m2(grid, x, y, sensor_range_m, fov_rad=2.0 * math.pi,
                     view_bearing=0.0, num_rays=72, step_m=0.0):
    """Estimate the unknown area visible from a viewpoint.

    Args:
        grid: GridMap holding the current occupancy grid.
        x: Viewpoint x in the map frame [m].
        y: Viewpoint y in the map frame [m].
        sensor_range_m: Sensor range [m].
        fov_rad: Horizontal field of view [rad]; capped at 2*pi.
        view_bearing: Direction the sensor faces in the map frame [rad];
            only relevant when fov_rad is below 2*pi.
        num_rays: Number of rays spread evenly over the field of view.
        step_m: Sample spacing along each ray [m]; 0 uses the map
            resolution, and values below half a cell are raised to it.

    Returns:
        Visible unknown area [m^2]; an optimistic upper bound (see the
        module docstring).
    """
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

    # A sample is occluded once any earlier sample on the same ray (row of
    # the array) was occupied; the running maximum propagates that along r.
    blocked = values >= grid.occupied_min
    occluded = np.maximum.accumulate(blocked, axis=1)

    visible_unknown = (values < 0) & ~occluded

    area_elements = radii * dtheta * step                # per sample on a ray
    return float((visible_unknown * area_elements[np.newaxis, :]).sum())
