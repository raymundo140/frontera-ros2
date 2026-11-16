"""
exploration_memory.py — unified spatial exploration memory.

One RegionRecord per frontier REGION (matched by distance, not by exact map
cell, because a frontier moves slightly as SLAM revises the map). A record
holds everything the framework has learned about that region:

  navigation failures      -> escalating cooldowns, then exhausted
  unproductive visits      -> utility penalty, then exhausted
  wavefront unreachability -> streak counter, probe result
  exhaustion snapshot      -> lets the region be reopened if the map around
                              it later changes

The memory also keeps the recently reached goals and the robot's recent
travel direction (from actual displacement, not chassis heading).

All times are behavior time (see timebase.py).
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegionRecord:
    """Everything learned about one frontier region.

    Attributes:
        x: Region center x in the map frame [m]; follows the frontier.
        y: Region center y in the map frame [m].
        cell_x: Grid column of the latest goal in this region.
        cell_y: Grid row of the latest goal in this region.
        failures: Navigation failures toward this region.
        unproductive: Reached goals that revealed too little new map.
        successes: Reserved; a productive visit removes the record instead.
        cooldown_until: Behavior time until which the region is skipped
            [s]; 0 when no cooldown is active.
        exhausted: True when the region is closed after repeated failures
            or unproductive visits.
        exhausted_reason: 'navigation_failures' or 'unproductive_visits'.
        unreachable_streak: Consecutive distinct maps on which the
            wavefront could not reach the region.
        unreachable_last_seq: Map sequence of the last unreachable count.
        unreachable_confirmed: True once a navigation probe confirmed that
            the region cannot be reached.
        probes_failed: Failed navigation probes toward the region.
        reopen_count: How often the region has been reopened.
        ref_known_cells: Known cells around the region when it was closed;
            the reference for deciding whether to reopen it.
    """

    x: float
    y: float
    cell_x: int = 0
    cell_y: int = 0
    failures: int = 0
    unproductive: int = 0
    successes: int = 0
    cooldown_until: float = 0.0
    exhausted: bool = False
    exhausted_reason: str = ''
    unreachable_streak: int = 0
    unreachable_last_seq: int = -1
    unreachable_confirmed: bool = False
    probes_failed: int = 0
    reopen_count: int = 0
    ref_known_cells: Optional[int] = None

    @property
    def closed(self):
        """True if the region is exhausted or confirmed unreachable."""
        return self.exhausted or self.unreachable_confirmed


class ExplorationMemory:
    """Unified spatial memory of exploration outcomes.

    Attributes:
        records: RegionRecords, one per remembered frontier region.
        failure_records: Navigation-failure records keyed by quantized goal
            position; used only by the 'exact_goal' ablation policy.
        recent_goals: Recently reached goals (x, y) in the map frame [m].
        preferred_direction: Unit vector of the robot's recent travel
            direction, or None before the robot has moved far enough.
    """

    def __init__(self, cfg):
        """Create an empty memory.

        Args:
            cfg: ExplorationConfig.
        """
        self.cfg = cfg
        self.records = []
        # Navigation-only failure states for the exact-goal ablation.
        # Unreachability and unproductive-visit memory remain spatial.
        self.failure_records = {}
        self.recent_goals = deque(maxlen=max(1, int(cfg.recent_goal_history_size)))
        self.preferred_direction = None
        self._travel_anchor = None

    # ── region lookup ────────────────────────────────────────────────────
    def find(self, x, y):
        """Return the region record nearest to (x, y), if close enough.

        Args:
            x: Point x in the map frame [m].
            y: Point y in the map frame [m].

        Returns:
            The nearest RegionRecord within region_match_radius_m, or None.
        """
        best = None
        best_d = float('inf')
        radius = self.cfg.region_match_radius_m
        for rec in self.records:
            d = math.hypot(float(x) - rec.x, float(y) - rec.y)
            if d <= radius and d < best_d:
                best = rec
                best_d = d
        return best

    def _failure_key(self, x, y):
        """Quantize a goal position to the exact-goal ablation's key."""
        resolution = float(self.cfg.failure_goal_quantization_m)
        return (math.floor(float(x) / resolution + 0.5),
                math.floor(float(y) / resolution + 0.5))

    def find_failure(self, x, y):
        """Return the record holding navigation failures for (x, y).

        Under the default 'region' policy this is the region record; under
        'exact_goal' it is the record of the exact (quantized) goal.
        """
        if self.cfg.failure_memory_policy == 'region':
            return self.find(x, y)
        return self.failure_records.get(self._failure_key(x, y))

    def _get_or_create_failure(self, goal):
        """Return the failure record for a goal, creating it if needed."""
        if self.cfg.failure_memory_policy == 'region':
            return self._get_or_create(goal)
        x, y = float(goal['world_x']), float(goal['world_y'])
        key = self._failure_key(x, y)
        if key not in self.failure_records:
            self.failure_records[key] = RegionRecord(x=x, y=y)
        return self.failure_records[key]

    def _all_memory_records(self):
        """Return region records plus any exact-goal failure records."""
        # Region and failure states are disjoint when exact_goal is enabled.
        return (self.records if self.cfg.failure_memory_policy == 'region'
                else self.records + list(self.failure_records.values()))

    def _get_or_create(self, goal):
        """Return the region record for a goal, creating it if needed."""
        rec = self.find(goal['world_x'], goal['world_y'])
        if rec is None:
            rec = RegionRecord(
                x=float(goal['world_x']), y=float(goal['world_y']))
            self.records.append(rec)
        # Keep the record centered near the latest version of the frontier.
        rec.x = 0.5 * rec.x + 0.5 * float(goal['world_x'])
        rec.y = 0.5 * rec.y + 0.5 * float(goal['world_y'])
        rec.cell_x = int(goal.get('cell_x', 0))
        rec.cell_y = int(goal.get('cell_y', 0))
        return rec

    def _remove(self, rec):
        """Forget a region record; no-op if it is already gone."""
        try:
            self.records.remove(rec)
        except ValueError:
            pass

    # ── outcomes ─────────────────────────────────────────────────────────
    def record_failure(self, goal, now, local_known_cells, probe=False):
        """Record a navigation failure toward a goal.

        Default (progressive) policy: each failure imposes the next, longer
        cooldown from failure_cooldowns_s, and failure number
        exhaust_after_failures exhausts the region. A failed probe instead
        counts toward confirming the region unreachable.

        Args:
        goal: Goal dictionary with at least 'world_x' and 'world_y' (map
            frame [m]); 'cell_x'/'cell_y' are stored when present.
            now: Behavior time of the failure [s].
            local_known_cells: Known cells around the goal now; stored as
                the reopening reference if the region is closed.
            probe: True if the goal was a probe toward a region that the
                wavefront considered unreachable.

        Returns:
            Dictionary describing the outcome for the event log, with
            'action' set to 'cooldown', 'exhausted' or
            'unreachable_confirmed'.
        """
        rec = self._get_or_create_failure(goal)
        rec.failures += 1
        info = {'failures': rec.failures, 'x': rec.x, 'y': rec.y}

        if probe:
            # Probe-unreachability remains a spatial mapping fact even in
            # exact-goal failure tracking. Do not disable reachability memory.
            probe_rec = (rec if self.cfg.failure_memory_policy == 'region'
                         else self._get_or_create(goal))
            probe_rec.probes_failed += 1
            if probe_rec.probes_failed >= max(1, self.cfg.unreachable_probe_limit):
                probe_rec.unreachable_confirmed = True
                probe_rec.ref_known_cells = int(local_known_cells)
                rec.cooldown_until = 0.0
                info['action'] = 'unreachable_confirmed'
                return info

        if self.cfg.failure_recovery_policy == 'fixed':
            duration = float(self.cfg.fixed_failure_cooldown_s)
            rec.cooldown_until = float(now) + duration
            info['action'] = 'cooldown'
            info['cooldown_s'] = duration
            return info

        if rec.failures >= self.cfg.exhaust_after_failures:
            rec.exhausted = True
            rec.exhausted_reason = 'navigation_failures'
            rec.cooldown_until = 0.0
            rec.ref_known_cells = int(local_known_cells)
            info['action'] = 'exhausted'
            return info

        cooldowns = self.cfg.failure_cooldowns_s
        duration = float(cooldowns[min(rec.failures - 1, len(cooldowns) - 1)])
        rec.cooldown_until = float(now) + duration
        info['action'] = 'cooldown'
        info['cooldown_s'] = duration
        return info

    def record_unproductive(self, goal, local_known_cells):
        """Record a reached goal that revealed too little new map.

        Each unproductive visit lowers the region's visit factor V; visit
        number unproductive_exhaust_after exhausts the region.

        Args:
        goal: Goal dictionary with at least 'world_x' and 'world_y' (map
            frame [m]); 'cell_x'/'cell_y' are stored when present.
            local_known_cells: Known cells around the goal now; stored as
                the reopening reference if the region is exhausted.

        Returns:
            Dictionary for the event log, with 'action' set to
            'penalized' or 'exhausted'.
        """
        rec = self._get_or_create(goal)
        rec.unproductive += 1
        info = {'unproductive': rec.unproductive, 'x': rec.x, 'y': rec.y}
        if rec.unproductive >= self.cfg.unproductive_exhaust_after:
            rec.exhausted = True
            rec.exhausted_reason = 'unproductive_visits'
            rec.cooldown_until = 0.0
            rec.ref_known_cells = int(local_known_cells)
            info['action'] = 'exhausted'
        else:
            info['action'] = 'penalized'
        return info

    def record_productive(self, goal):
        """Record a productive visit: forget the region's history.

        Args:
        goal: Goal dictionary with at least 'world_x' and 'world_y' (map
            frame [m]); 'cell_x'/'cell_y' are stored when present.
        """
        rec = self.find(goal['world_x'], goal['world_y'])
        if rec is not None:
            self._remove(rec)
        if self.cfg.failure_memory_policy == 'exact_goal':
            self.failure_records.pop(
                self._failure_key(goal['world_x'], goal['world_y']), None)

    def record_reached(self, x, y):
        """Add a reached goal (map frame [m]) to the recent-goal history."""
        self.recent_goals.append((float(x), float(y)))

    # ── reachability bookkeeping ─────────────────────────────────────────
    def note_unreachable(self, goal, map_seq):
        """Count one more distinct map on which the region is cut off.

        Repeated calls with the same map_seq count only once, so the streak
        measures consecutive distinct maps.

        Args:
        goal: Goal dictionary with at least 'world_x' and 'world_y' (map
            frame [m]); 'cell_x'/'cell_y' are stored when present.
            map_seq: Sequence number of the current map.

        Returns:
            The region's RegionRecord.
        """
        rec = self._get_or_create(goal)
        if rec.unreachable_last_seq != int(map_seq):
            rec.unreachable_streak += 1
            rec.unreachable_last_seq = int(map_seq)
        return rec

    def note_reachable(self, rec):
        """Reset the unreachable streak of a region the wavefront reaches.

        A record that holds no other information is removed.

        Args:
            rec: RegionRecord, or None (no-op).
        """
        if rec is None:
            return
        rec.unreachable_streak = 0
        rec.unreachable_last_seq = -1
        if (not rec.closed and rec.failures == 0 and rec.unproductive == 0
                and rec.cooldown_until <= 0.0):
            self._remove(rec)           # nothing left worth remembering

    # ── reopening ────────────────────────────────────────────────────────
    def can_reopen(self, rec):
        """True if the region has reopens left (max_reopens_per_region)."""
        return rec.reopen_count < self.cfg.max_reopens_per_region

    def reopen(self, rec):
        """Reopen a closed region for one more attempt.

        The failure and unproductive counters are set one below their
        exhaustion thresholds, so a single further failure or unproductive
        visit closes the region again.

        Args:
            rec: The RegionRecord to reopen.
        """
        rec.reopen_count += 1
        rec.exhausted = False
        rec.exhausted_reason = ''
        rec.unreachable_confirmed = False
        rec.unreachable_streak = 0
        rec.unreachable_last_seq = -1
        rec.probes_failed = 0
        rec.cooldown_until = 0.0
        rec.ref_known_cells = None
        rec.failures = max(0, self.cfg.exhaust_after_failures - 1)
        rec.unproductive = max(0, self.cfg.unproductive_exhaust_after - 1)

    def reopen_changed_regions(self, grid):
        """Reopen closed regions whose surroundings gained enough new map.

        A region is reopened when at least reopen_min_new_area_m2 of new
        known map appeared within reopen_radius_m of it since it closed,
        for example because a temporary obstruction moved away.

        Args:
            grid: GridMap of the current map.

        Returns:
            List of (x, y, gained area [m^2]) for every reopened region.
        """
        reopened = []
        cell_area = grid.resolution * grid.resolution
        for rec in self._all_memory_records():
            if not rec.closed or rec.ref_known_cells is None:
                continue
            if not self.can_reopen(rec):
                continue
            known = grid.count_known_in_window(
                rec.x, rec.y, self.cfg.reopen_radius_m)
            gained = (known - rec.ref_known_cells) * cell_area
            if gained >= self.cfg.reopen_min_new_area_m2:
                self.reopen(rec)
                reopened.append((rec.x, rec.y, gained))
        return reopened

    # ── cooldowns ────────────────────────────────────────────────────────
    def in_cooldown(self, rec, now):
        """True if an open region's cooldown has not yet expired.

        Args:
            rec: RegionRecord, or None.
            now: Current behavior time [s].
        """
        return (
            rec is not None and not rec.closed and
            rec.cooldown_until > float(now)
        )

    def consume_expired_cooldowns(self, now):
        """Clear cooldowns that have expired.

        Args:
            now: Current behavior time [s].

        Returns:
            Number of cooldowns that expired since the last call.
        """
        expired = 0
        for rec in self._all_memory_records():
            if (not rec.closed and rec.cooldown_until > 0.0
                    and rec.cooldown_until <= float(now)):
                rec.cooldown_until = 0.0
                expired += 1
        return expired

    def active_cooldowns(self, now):
        """Return the open records whose cooldown is still running."""
        return [
            rec for rec in self._all_memory_records()
            if not rec.closed and rec.cooldown_until > float(now)
        ]

    def count_exhausted(self):
        """Return the number of exhausted records."""
        return sum(1 for rec in self._all_memory_records() if rec.exhausted)

    def count_unreachable(self):
        """Return the number of regions confirmed unreachable."""
        return sum(1 for rec in self.records if rec.unreachable_confirmed)

    # ── scoring factors ──────────────────────────────────────────────────
    def recent_factor(self, x, y):
        """Return the recent-goal factor R for a candidate.

        Args:
            x: Candidate x in the map frame [m].
            y: Candidate y in the map frame [m].

        Returns:
            recent_goal_penalty if the candidate lies within
            recent_goal_radius_m of a recently reached goal, else 1.0.
        """
        if not self.recent_goals:
            return 1.0
        nearest = min(
            math.hypot(float(x) - gx, float(y) - gy)
            for gx, gy in self.recent_goals)
        if nearest < self.cfg.recent_goal_radius_m:
            return float(self.cfg.recent_goal_penalty)
        return 1.0

    def visit_factor(self, rec):
        """Return the unproductive-visit factor V for a region.

        Args:
            rec: RegionRecord of the candidate, or None.

        Returns:
            unproductive_visit_penalty ** (unproductive visits); 1.0 for
            a region without unproductive visits.
        """
        if rec is None or rec.unproductive <= 0:
            return 1.0
        return float(self.cfg.unproductive_visit_penalty) ** rec.unproductive

    # ── travel direction from actual displacement ────────────────────────
    def reset_travel_anchor(self):
        """Restart displacement tracking from the next pose update."""
        self._travel_anchor = None

    def update_travel(self, x, y):
        """Update the preferred travel direction from the robot position.

        Once the robot has moved at least direction_min_update_distance_m
        from the last anchor, the direction of that displacement is blended
        into the preferred direction (weight direction_memory_keep on the
        old one). Using actual displacement rather than chassis heading
        keeps this correct for omnidirectional bases.

        Args:
            x: Robot x in the map frame [m].
            y: Robot y in the map frame [m].
        """
        if self._travel_anchor is None:
            self._travel_anchor = (float(x), float(y))
            return
        dx = float(x) - self._travel_anchor[0]
        dy = float(y) - self._travel_anchor[1]
        length = math.hypot(dx, dy)
        if length < self.cfg.direction_min_update_distance_m:
            return

        new_dir = (dx / length, dy / length)
        if self.preferred_direction is None:
            blended = new_dir
        else:
            keep = self.cfg.direction_memory_keep
            mx = keep * self.preferred_direction[0] + (1.0 - keep) * new_dir[0]
            my = keep * self.preferred_direction[1] + (1.0 - keep) * new_dir[1]
            norm = math.hypot(mx, my)
            blended = (mx / norm, my / norm) if norm > 1e-9 else new_dir

        self.preferred_direction = blended
        self._travel_anchor = (float(x), float(y))
