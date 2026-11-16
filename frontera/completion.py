"""
completion.py — map-progress monitoring and exploration-end criteria.

MapProgress is updated on EVERY received occupancy grid (also while the robot
is navigating), so map growth is not sampled only at decision time.

CompletionMonitor decides when exploration ends. It never ends a mission just
because no destination is selectable right now:

  Criterion A — frontier exhaustion  (COMPLETE / COMPLETE_PARTIAL)
    No candidate that could still be explored (ELIGIBLE, COOLDOWN or
    TEMPORARILY_BLOCKED) on `completion_confirm_maps` consecutive NEW maps
    (or idle re-evaluations, if the SLAM system stops republishing the map).
      COMPLETE          only low-gain / unsafe remnants are left
      COMPLETE_PARTIAL  exhausted or unreachable frontiers remain

  Criterion B — stagnation  (STAGNATED)
    Candidates that could be explored still exist, but nothing has been
    selectable and no goal has been active for `stagnation_timeout_s` of
    running time, AND the map grew by less than `stagnation_min_growth_m2`
    over the progress window. Low map growth alone never ends a mission.

TIME_LIMIT (max_exploration_time_s reached) and ERROR_NAV_UNRESPONSIVE (the
navigation server stopped answering) are decided by the controller; the
names are defined here so that all end classes live in one place.
"""

from collections import deque

from .grid import map_change_pct
from . import frontier_scoring as fs

COMPLETE = 'COMPLETE'
COMPLETE_PARTIAL = 'COMPLETE_PARTIAL'
STAGNATED = 'STAGNATED'
TIME_LIMIT = 'TIME_LIMIT'
ERROR_NAV_UNRESPONSIVE = 'ERROR_NAV_UNRESPONSIVE'


class MapProgress:
    """Tracks known-map area and map change over time.

    Attributes:
        maps_received: Number of maps received so far.
        known_area_m2: Known area of the latest map [m^2].
        new_area_last_m2: Area change from the previous map [m^2].
        growth_m2_min: Known-area growth rate over the last 60 s
            [m^2/min], for status reporting only.
        last_change_pct: Share of known cells changed since the previous
            map [%], or None for the first map.
        stable_maps: Consecutive maps whose change stayed below
            map_stable_change_threshold_pct.
        last_update_t: Behavior time of the latest map [s].
    """

    # Window of the reported growth rate [s], independent of the configured
    # progress window used for stagnation.
    LEGACY_RATE_WINDOW_S = 60.0

    def __init__(self, cfg):
        """Create an empty progress tracker.

        Args:
            cfg: ExplorationConfig.
        """
        self.cfg = cfg
        self._history = deque()
        self._previous = None

        self.maps_received = 0
        self.known_area_m2 = 0.0
        self.new_area_last_m2 = 0.0
        self.growth_m2_min = 0.0
        self.last_change_pct = None
        self.stable_maps = 0
        self.last_update_t = None

    def update(self, grid, now):
        """Record a newly received map.

        Args:
            grid: GridMap of the new map.
            now: Behavior time of reception [s].
        """
        self.maps_received += 1
        area = grid.known_area_m2
        self.new_area_last_m2 = area - self.known_area_m2
        self.known_area_m2 = area
        self.last_update_t = float(now)

        change = map_change_pct(self._previous, grid)
        self._previous = grid
        self.last_change_pct = change
        if change is None:
            self.stable_maps = 0
        elif change < self.cfg.map_stable_change_threshold_pct:
            self.stable_maps += 1
        else:
            self.stable_maps = 0

        self._history.append((float(now), area))
        keep = max(self.cfg.map_progress_window_s, self.LEGACY_RATE_WINDOW_S)
        while (len(self._history) >= 2
               and float(now) - self._history[0][0] > keep):
            self._history.popleft()

        # Dashboard rate over (up to) the last 60 s.
        old_t, old_area = self._sample_at(float(now) - self.LEGACY_RATE_WINDOW_S)
        dt = float(now) - old_t
        self.growth_m2_min = (
            (area - old_area) * 60.0 / dt if dt > 1e-6 else 0.0)

    def _sample_at(self, t):
        """Return the oldest stored (time, area) sample not older than t."""
        for sample in self._history:
            if sample[0] >= t:
                return sample
        return self._history[-1]

    def growth_over_window(self, now):
        """Return the known-area growth over the progress window.

        Args:
            now: Current behavior time [s].

        Returns:
            Growth of the known area over the last map_progress_window_s
            [m^2]; 0.0 before the first map.
        """
        if not self._history:
            return 0.0
        _, old_area = self._sample_at(
            float(now) - self.cfg.map_progress_window_s)
        return self.known_area_m2 - old_area


class CompletionMonitor:
    """Decides on evidence when exploration is over.

    The controller calls evaluate() once per decision cycle with the
    current count of candidates per class (see frontier_scoring).

    Attributes:
        no_candidate_maps: Consecutive new maps (or idle re-evaluations)
            on which no explorable candidate was found.
    """

    def __init__(self, cfg):
        """Create a monitor with no evidence collected yet.

        Args:
            cfg: ExplorationConfig.
        """
        self.cfg = cfg
        self.no_candidate_maps = 0
        self._last_map_seq = -1
        self._last_activity = 0.0
        self._last_count_time = 0.0

    def note_activity(self, run_time):
        """Record that a goal was dispatched or finished.

        Resets the stagnation timer, since exploration is evidently active.

        Args:
            run_time: Running exploration time [s] (excludes pauses).
        """
        self._last_activity = float(run_time)

    def evaluate(self, counts, map_seq, run_time, window_growth_m2,
                 selectable=False):
        """Decide whether exploration has ended.

        Args:
            counts: Dictionary {candidate class: number of candidates} for
                the current decision cycle.
            map_seq: Sequence number of the map the counts come from.
            run_time: Running exploration time [s] (excludes pauses).
            window_growth_m2: Known-area growth over the progress window
                [m^2], from MapProgress.growth_over_window().
            selectable: True if a candidate (including an unreachability
                probe) could be dispatched now; used only by the
                single_no_eligible ablation policy.

        Returns:
            None while exploration continues; otherwise a dictionary
            {'class': end class, 'reason': human-readable explanation}.
        """
        if self.cfg.completion_policy == 'single_no_eligible':
            # Ablation policy: end on the first cycle without a selectable
            # frontier. The controller only calls this with a valid map and
            # a fresh robot pose, and it never ends while a probe is
            # selectable. Cooling and blocked frontiers are deliberately
            # ignored, to measure premature completion against the full
            # evidence-based policy.
            if not selectable and counts.get(fs.ELIGIBLE, 0) == 0:
                return {
                    'class': COMPLETE,
                    'reason': 'A5_single_evaluation_no_selectable_frontier',
                }
            return None
        blocking = sum(counts.get(c, 0) for c in fs.BLOCKING_CLASSES)

        if counts.get(fs.ELIGIBLE, 0) > 0:
            self._last_activity = float(run_time)

        period = self.cfg.idle_reevaluation_period_s
        if blocking > 0:
            self.no_candidate_maps = 0
            self._last_count_time = float(run_time)
        elif int(map_seq) != self._last_map_seq:
            # Advance at most once per NEW occupancy grid.
            self.no_candidate_maps += 1
            self._last_count_time = float(run_time)
        elif period > 0.0 and float(run_time) - self._last_count_time >= period:
            # Some SLAM systems do not republish an unchanged map. Without
            # this, a finished exploration would wait for a map forever.
            self.no_candidate_maps += 1
            self._last_count_time = float(run_time)
        self._last_map_seq = int(map_seq)

        if self.no_candidate_maps >= self.cfg.completion_confirm_maps:
            unreached = (
                counts.get(fs.EXHAUSTED, 0) + counts.get(fs.UNREACHABLE, 0))
            if unreached > 0:
                return {
                    'class': COMPLETE_PARTIAL,
                    'reason': (
                        f'No explorable frontier on '
                        f'{self.cfg.completion_confirm_maps} consecutive new '
                        f'maps; {counts.get(fs.EXHAUSTED, 0)} exhausted and '
                        f'{counts.get(fs.UNREACHABLE, 0)} unreachable '
                        f'frontier regions remain'),
                }
            return {
                'class': COMPLETE,
                'reason': (
                    f'No explorable frontier on '
                    f'{self.cfg.completion_confirm_maps} consecutive new maps'),
            }

        idle_for = float(run_time) - self._last_activity
        if (blocking > 0
                and idle_for >= self.cfg.stagnation_timeout_s
                and window_growth_m2 < self.cfg.stagnation_min_growth_m2):
            return {
                'class': STAGNATED,
                'reason': (
                    f'No selectable frontier and no active goal for '
                    f'{idle_for:.0f} s while the map grew '
                    f'{window_growth_m2:.2f} m2; '
                    f'{counts.get(fs.COOLDOWN, 0)} cooling down and '
                    f'{counts.get(fs.TEMPORARILY_BLOCKED, 0)} temporarily '
                    f'blocked frontier regions remain'),
            }
        return None
