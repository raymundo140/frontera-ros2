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
        return self.exhausted or self.unreachable_confirmed


class ExplorationMemory:

    def __init__(self, cfg):
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
        resolution = float(self.cfg.failure_goal_quantization_m)
        return (math.floor(float(x) / resolution + 0.5),
                math.floor(float(y) / resolution + 0.5))

    def find_failure(self, x, y):
        if self.cfg.failure_memory_policy == 'region':
            return self.find(x, y)
        return self.failure_records.get(self._failure_key(x, y))

    def _get_or_create_failure(self, goal):
        if self.cfg.failure_memory_policy == 'region':
            return self._get_or_create(goal)
        x, y = float(goal['world_x']), float(goal['world_y'])
        key = self._failure_key(x, y)
        if key not in self.failure_records:
            self.failure_records[key] = RegionRecord(x=x, y=y)
        return self.failure_records[key]

    def _all_memory_records(self):
        # Region and failure states are disjoint when exact_goal is enabled.
        return (self.records if self.cfg.failure_memory_policy == 'region'
                else self.records + list(self.failure_records.values()))

    def _get_or_create(self, goal):
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
        try:
            self.records.remove(rec)
        except ValueError:
            pass

    # ── outcomes ─────────────────────────────────────────────────────────
    def record_failure(self, goal, now, local_known_cells, probe=False):
        """Escalating policy: cooldown, longer cooldown, ..., exhausted."""
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
        """A visit that revealed new map clears the region's history."""
        rec = self.find(goal['world_x'], goal['world_y'])
        if rec is not None:
            self._remove(rec)
        if self.cfg.failure_memory_policy == 'exact_goal':
            self.failure_records.pop(
                self._failure_key(goal['world_x'], goal['world_y']), None)

    def record_reached(self, x, y):
        self.recent_goals.append((float(x), float(y)))

    # ── reachability bookkeeping ─────────────────────────────────────────
    def note_unreachable(self, goal, map_seq):
        """Count consecutive DISTINCT maps on which the region is cut off."""
        rec = self._get_or_create(goal)
        if rec.unreachable_last_seq != int(map_seq):
            rec.unreachable_streak += 1
            rec.unreachable_last_seq = int(map_seq)
        return rec

    def note_reachable(self, rec):
        if rec is None:
            return
        rec.unreachable_streak = 0
        rec.unreachable_last_seq = -1
        if (not rec.closed and rec.failures == 0 and rec.unproductive == 0
                and rec.cooldown_until <= 0.0):
            self._remove(rec)           # nothing left worth remembering

    # ── reopening ────────────────────────────────────────────────────────
    def can_reopen(self, rec):
        return rec.reopen_count < self.cfg.max_reopens_per_region

    def reopen(self, rec):
        """Give a closed region one more attempt before it closes again."""
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
        """Reopen closed regions whose surroundings gained enough new map."""
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
        return (
            rec is not None and not rec.closed and
            rec.cooldown_until > float(now)
        )

    def consume_expired_cooldowns(self, now):
        """Clear expired cooldowns once; return how many expired."""
        expired = 0
        for rec in self._all_memory_records():
            if (not rec.closed and rec.cooldown_until > 0.0
                    and rec.cooldown_until <= float(now)):
                rec.cooldown_until = 0.0
                expired += 1
        return expired

    def active_cooldowns(self, now):
        return [
            rec for rec in self._all_memory_records()
            if not rec.closed and rec.cooldown_until > float(now)
        ]

    def count_exhausted(self):
        return sum(1 for rec in self._all_memory_records() if rec.exhausted)

    def count_unreachable(self):
        return sum(1 for rec in self.records if rec.unreachable_confirmed)

    # ── scoring factors ──────────────────────────────────────────────────
    def recent_factor(self, x, y):
        if not self.recent_goals:
            return 1.0
        nearest = min(
            math.hypot(float(x) - gx, float(y) - gy)
            for gx, gy in self.recent_goals)
        if nearest < self.cfg.recent_goal_radius_m:
            return float(self.cfg.recent_goal_penalty)
        return 1.0

    def visit_factor(self, rec):
        if rec is None or rec.unproductive <= 0:
            return 1.0
        return float(self.cfg.unproductive_visit_penalty) ** rec.unproductive

    # ── travel direction from actual displacement ────────────────────────
    def reset_travel_anchor(self):
        self._travel_anchor = None

    def update_travel(self, x, y):
        """Blend the preferred direction with the latest real displacement."""
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
