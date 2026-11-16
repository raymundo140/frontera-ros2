"""
controller.py — the exploration decision loop (no ROS imports).

The controller owns the control state machine (WAITING / RUNNING / PAUSED /
STOPPED), the single frontier-selection gate, the spatial memory and the
completion decision. Executing one selected goal is delegated to
NavigationSupervisor.

It talks to the outside world only through a `ports` object supplied by the
ROS node (or by a test double). Required port methods:

    ros_now() -> float            ROS time [s]
    wall_now() -> float           monotonic wall time [s]
    get_pose() -> (x, y, yaw)|None   fresh map-frame robot pose
    reset_tf()                    drop buffered transforms (clock reset)
    nav_ready() -> bool
    nav_send(txn_id, x, y, yaw) -> bool
    nav_cancel(handle)
    cmd_rotate(angular_z)
    publish_goal(x, y, yaw)
    publish_status(text)
    publish_stats(text)
    publish_event(json_text)
    publish_markers(list_of_dicts)
    point_sensor(bearing_rad)
    request_map_save()
    publish_map_save_status(text)
    log(level, text)

Decision pipeline, executed only from tick() and only when no transaction
is active:

    cached map -> frontier cells -> clusters -> reachability wavefront ->
    safe goal cell per cluster -> information gain -> memory lookup ->
    classification -> utility -> best candidate -> NavigationSupervisor
"""

import json
import math
import time

import numpy as np

from . import completion as comp
from . import frontier_scoring as fs
from . import navigation_supervisor as ns
from .completion import CompletionMonitor, MapProgress
from .exploration_memory import ExplorationMemory
from .frontier_detection import cluster_frontiers, detect_frontier_mask
from .information_gain import estimate_gain_m2, reference_gain_m2
from .navigation_supervisor import NavigationSupervisor
from .reachability import compute_reachability
from .timebase import BehaviorClock

STATE_WAITING = 'WAITING'
STATE_RUNNING = 'RUNNING'
STATE_PAUSED = 'PAUSED'
STATE_STOPPED = 'STOPPED'

EVENT_SCHEMA = 'exploration_event/1'
FRAMEWORK_VERSION = 2

# Visit classifications (after the post-arrival map check).
PRODUCTIVE_SUCCESS = 'PRODUCTIVE_SUCCESS'
UNPRODUCTIVE_SUCCESS = 'UNPRODUCTIVE_SUCCESS'
UNVERIFIED_SUCCESS = 'UNVERIFIED_SUCCESS'
NAVIGATION_FAILURE = 'NAVIGATION_FAILURE'
FRONTIER_RESOLVED = 'FRONTIER_RESOLVED'
INTENTIONAL_CANCELLATION = 'INTENTIONAL_CANCELLATION'


def _num(value, digits=4):
    """JSON-safe number: None for NaN / inf / missing."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


class ExplorationController:

    def __init__(self, cfg, ports):
        for warning in cfg.validate():
            ports.log('warn', f'Configuration: {warning}')

        self.cfg = cfg
        self.p = ports

        self.clock = BehaviorClock(
            cfg.clock_max_forward_step_s, cfg.frozen_clock_wall_timeout_s)
        self.memory = ExplorationMemory(cfg)
        self.progress = MapProgress(cfg)
        self.completion = CompletionMonitor(cfg)
        self.supervisor = NavigationSupervisor(
            cfg, ports,
            motion_allowed=self._motion_allowed,
            on_finished=self._on_transaction_finished,
            emit=self._emit,
            rotation_clear=self._rotation_clear,
        )

        # control
        self.ctrl_state = STATE_WAITING
        self.ended = False
        self.end_class = ''
        self.completion_reason = ''
        self.needs_selection = False
        self._started = False
        self._pending_end = None
        self._settle = None
        self._idle_reason = ''
        self._frozen = False

        # data
        self.grid = None
        self.costmap = None
        self._map_seq = 0
        self._costmap_frame_warned = False

        # time accounting
        self._last_tick_t = None
        self._last_select_t = 0.0
        self._run_time = 0.0

        # statistics
        self.goals_reached = 0
        self.goals_failed = 0
        self.goals_resolved = 0
        self.goals_canceled = 0
        self.visits_productive = 0
        self.visits_unproductive = 0
        self.dist_travelled_m = 0.0
        self._last_pose_xy = None
        self._last_pose = None
        self._event_seq = 0
        self._throttle = {}

        # diagnostics of the latest decision
        self.frontier_count = 0
        self.frontier_mode = 'none'
        self.frontier_full = 0
        self.frontier_ms = 0.0
        self._last_counts = {c: 0 for c in fs.ALL_CLASSES}
        self._last_timing = {
            'detect_ms': 0.0, 'cluster_ms': 0.0, 'reach_ms': 0.0,
            'gain_ms': 0.0, 'score_ms': 0.0, 'decision_ms': 0.0}
        self._map_safety_rejected = 0
        self._costmap_safety_rejected = 0
        self._edge_rejected = 0
        self._small_rejected = 0

    # =====================================================================
    # helpers
    # =====================================================================
    def _now(self):
        wall = self.p.wall_now()
        t = self.clock.update(self.p.ros_now(), wall)
        if self.clock.consume_reset():
            # Buffered transforms are stamped in the old time line.
            self.p.reset_tf()
            self._emit('CLOCK_JUMP', {
                'backward_jumps': self.clock.backward_jumps})
            self.p.log(
                'warn',
                'ROS clock jumped backward; deadlines keep their remaining '
                'duration and the TF buffer was cleared')
        return t, wall

    def _motion_allowed(self):
        return self.ctrl_state == STATE_RUNNING and not self.ended

    def _rotation_clear(self, x, y, radius_m):
        if self.grid is None:
            return False
        return self.grid.is_clear(x, y, radius_m)

    def _log_throttled(self, key, period_s, level, text):
        wall = self.p.wall_now()
        if wall - self._throttle.get(key, -1e18) >= period_s:
            self._throttle[key] = wall
            self.p.log(level, text)

    def _emit(self, event, data=None, txn_id=None):
        self._event_seq += 1
        payload = {
            'schema': EVENT_SCHEMA,
            'seq': self._event_seq,
            'event': event,
            't_ros': _num(self.p.ros_now(), 3),
            't_behavior': _num(self.clock.t, 3),
            't_run': _num(self._run_time, 3),
            't_unix': _num(time.time(), 3),
            'txn_id': int(txn_id) if txn_id else self.supervisor.txn_id,
            'state': self.ctrl_state,
            'data': data or {},
        }
        try:
            self.p.publish_event(json.dumps(payload, separators=(',', ':')))
        except Exception as exc:            # never let diagnostics break control
            self.p.log('warn', f'event publish failed: {exc}')

    # =====================================================================
    # inputs
    # =====================================================================
    def on_map(self, grid):
        now, wall = self._now()
        self._map_seq += 1
        grid.seq = self._map_seq
        self.grid = grid
        self.progress.update(grid, now)

        if self.supervisor.phase == ns.NAVIGATING:
            self._reevaluate_active_goal(grid, now, wall)

        if (self._started and self._motion_allowed()
                and not self.supervisor.busy and self._settle is None):
            # A new occupancy grid allows a genuine new frontier refresh.
            self.needs_selection = True

    def on_costmap(self, grid):
        self.costmap = grid

    def on_command(self, text):
        cmd = str(text).strip().upper()
        now, wall = self._now()

        if cmd == 'START':
            if self.ended:
                self.p.log(
                    'warn',
                    f'Exploration already ended ({self.end_class}). '
                    'Relaunch the node for a new run.')
            elif self.ctrl_state in (STATE_WAITING, STATE_PAUSED):
                self.ctrl_state = STATE_RUNNING
                self.needs_selection = True
                self.completion.note_activity(self._run_time)
                self._emit('CONTROL', {'command': 'START'})
                self.p.log('info', 'Exploration STARTED')
            elif self.ctrl_state == STATE_RUNNING:
                self.p.log('info', 'Already running')
            else:
                self.p.log(
                    'warn', 'Cannot restart: exploration is STOPPED. '
                    'Relaunch the node.')

        elif cmd == 'PAUSE':
            if self.ctrl_state == STATE_RUNNING and not self.ended:
                # State first: from here every motion path is gated off.
                self.ctrl_state = STATE_PAUSED
                self.needs_selection = False
                self._emit('CONTROL', {'command': 'PAUSE'})
                self.p.log('info', 'Exploration PAUSED')
                self.supervisor.request_cancel('pause', now, wall)
            else:
                self.p.log(
                    'info', f'Cannot pause: state={self.ctrl_state}, '
                    f'ended={self.ended}')

        elif cmd == 'STOP':
            if self.ctrl_state != STATE_STOPPED:
                # STOP is terminal for this node instance. Process shutdown
                # is the program manager's job, not this node's.
                self.ctrl_state = STATE_STOPPED
                self.needs_selection = False
                self._emit('CONTROL', {'command': 'STOP'})
                self.p.log('info', 'Exploration STOPPED')
                self.supervisor.request_cancel('stop', now, wall)
                self.p.publish_status(self.status_text())
            else:
                self.p.log('info', 'Already stopped')

        elif cmd == 'SAVE_MAP':
            self._save_map()

        elif cmd == 'RESET_NAV_FAULT':
            if self.supervisor.phase == ns.FAULT:
                # An unconfirmed goal may still be moving. No control message
                # may release it; the operator must restart the entire stack
                # if a terminal result never arrives.
                self.supervisor.acknowledge_fault(now)
            else:
                self.p.log('info', 'No active navigation fault')

        else:
            self.p.log('warn', f'Unknown control command: {text}')

    # navigation callbacks: forwarded with their transaction ID
    def nav_goal_response(self, txn_id, accepted, handle, error=None):
        now, wall = self._now()
        self.supervisor.on_goal_response(
            txn_id, accepted, handle, error, now, wall)

    def nav_feedback(self, txn_id, distance_remaining, recoveries):
        self.supervisor.on_feedback(
            txn_id, distance_remaining, recoveries, self.clock.t)

    def nav_result(self, txn_id, status, error=None):
        now, _ = self._now()
        self.supervisor.on_result(txn_id, status, error, now)

    # =====================================================================
    # timers
    # =====================================================================
    def rotation_tick(self):
        if self.supervisor.phase != ns.ROTATING:
            return
        now, wall = self._now()
        self.supervisor.rotation_tick(now, wall)

    def wall_tick(self):
        """Runs on a wall-clock timer, so it fires even if ROS time stops."""
        now, wall = self._now()
        frozen = self.clock.is_frozen(wall)
        if frozen and not self._frozen:
            self._frozen = True
            self.supervisor.halt_direct_motion()
            self._emit('CLOCK_FROZEN', {'phase': self.supervisor.phase})
            if self.ctrl_state == STATE_RUNNING:
                self.p.log(
                    'warn', 'ROS clock is not advancing; behavior timers '
                    'are suspended until it resumes')
        elif not frozen and self._frozen:
            self._frozen = False
            self._emit('CLOCK_RESUMED', {})
        # Wall-clock bounds of SENDING / CANCELING must be checked even when
        # the ROS-time timers are not firing.
        self.supervisor.tick(now, wall)

    def status_tick(self):
        self._now()
        self.p.publish_status(self.status_text())
        self.p.publish_stats(self.stats_text())

    def tick(self):
        """The single supervisory gate (decision_rate_hz)."""
        now, wall = self._now()
        dt = 0.0 if self._last_tick_t is None else max(0.0, now - self._last_tick_t)
        self._last_tick_t = now

        self.supervisor.tick(now, wall)

        pose = self.p.get_pose()
        self._track_pose(pose)

        if self.ended:
            return
        if self.ctrl_state == STATE_RUNNING:
            self._run_time += dt

        if self.supervisor.phase == ns.FAULT:
            limit = self.cfg.nav_fault_abort_timeout_s
            if limit > 0.0 and now - self.supervisor.fault_since >= limit:
                self._end(
                    comp.ERROR_NAV_UNRESPONSIVE,
                    f'Navigation stack unresponsive for {limit:.0f} s '
                    f'({self.supervisor.fault_reason})')
            return

        if self._pending_end is not None:
            if not self.supervisor.busy:
                end_class, reason = self._pending_end
                self._pending_end = None
                self._end(end_class, reason)
            return

        limit = self.cfg.max_exploration_time_s
        if (limit > 0.0 and self.ctrl_state == STATE_RUNNING
                and self._run_time >= limit):
            self._pending_end = (
                comp.TIME_LIMIT,
                f'Exploration time limit of {limit:.0f} s reached')
            self.needs_selection = False
            self.supervisor.request_cancel('time_limit', now, wall)
            return

        self._process_settle(now)

        if self.memory.consume_expired_cooldowns(now):
            if self._motion_allowed():
                self.needs_selection = True

        if self.ctrl_state != STATE_RUNNING:
            return

        if not self._started:
            if self.grid is None or pose is None:
                self._log_throttled(
                    'wait_data', 2.0, 'info',
                    f'Waiting... map={"ok" if self.grid is not None else "no"} '
                    f'pose={"ok" if pose is not None else "no"}')
                return
            if not self.p.nav_ready():
                self._log_throttled(
                    'wait_nav', 2.0, 'info', 'Waiting for navigation action')
                return
            self.p.log('info', 'Data ready: first frontier search')
            self._started = True
            self.needs_selection = True

        if self.supervisor.busy or self._settle is not None:
            return
        period = self.cfg.idle_reevaluation_period_s
        if (not self.needs_selection and period > 0.0
                and now - self._last_select_t >= period):
            # Idle with nothing selectable: look again even without a new
            # map (the costmap, the pose or a cooldown may have changed).
            self.needs_selection = True
        if not self.needs_selection:
            return
        if not self.p.nav_ready():
            return                      # silently wait

        self.needs_selection = False
        self._last_select_t = now
        self._select(now, wall, pose)

    # =====================================================================
    # pose tracking (dashboard distance + travel direction)
    # =====================================================================
    def _track_pose(self, pose):
        self._last_pose = pose
        if pose is None:
            return
        x, y = pose[0], pose[1]
        if self._last_pose_xy is not None:
            step = math.hypot(x - self._last_pose_xy[0],
                              y - self._last_pose_xy[1])
            if step <= self.cfg.pose_jump_reject_m:
                self.dist_travelled_m += step
            else:
                # Localization correction, not motion.
                self.memory.reset_travel_anchor()
        self._last_pose_xy = (x, y)

        if self.supervisor.phase == ns.NAVIGATING:
            self.memory.update_travel(x, y)
        else:
            self.memory.reset_travel_anchor()

    # =====================================================================
    # frontier re-evaluation while navigating
    # =====================================================================
    def _goal_gain(self, grid, goal_x, goal_y, bearing):
        return estimate_gain_m2(
            grid, goal_x, goal_y,
            self.cfg.sensor_range_m,
            math.radians(self.cfg.sensor_fov_deg),
            bearing,
            self.cfg.gain_num_rays,
            self.cfg.gain_ray_step_m)

    def _reevaluate_active_goal(self, grid, now, wall):
        txn = self.supervisor.txn
        if txn is None:
            return
        goal = txn.goal
        try:
            gain = self._goal_gain(
                grid, goal['world_x'], goal['world_y'],
                goal.get('view_bearing', 0.0))
        except Exception as exc:
            self._log_throttled(
                'reeval', 10.0, 'warn', f'goal re-evaluation failed: {exc}')
            return
        threshold = max(
            self.cfg.min_information_gain_m2,
            self.cfg.resolved_gain_fraction * float(goal.get('gain_m2', 0.0)))
        goal['gain_now_m2'] = gain
        self.supervisor.vote_resolved(gain < threshold, now, wall)

    # =====================================================================
    # selection
    # =====================================================================
    def _costmap_ok(self, xs, ys):
        """Costmap VETO: reject only explicit high-cost cells."""
        cm = self.costmap
        if cm is None:
            return np.ones(xs.shape, dtype=bool)
        if self.grid is not None and cm.frame_id != self.grid.frame_id:
            if not self._costmap_frame_warned:
                self._costmap_frame_warned = True
                self.p.log(
                    'warn',
                    f'Costmap frame "{cm.frame_id}" differs from map frame '
                    f'"{self.grid.frame_id}"; costmap veto disabled')
            return np.ones(xs.shape, dtype=bool)

        cols = np.floor((xs - cm.origin_x) / cm.resolution).astype(np.int64)
        rows = np.floor((ys - cm.origin_y) / cm.resolution).astype(np.int64)
        inside = (
            (rows >= 0) & (rows < cm.height) &
            (cols >= 0) & (cols < cm.width))
        ok = np.ones(xs.shape, dtype=bool)
        values = cm.grid[rows[inside], cols[inside]]
        # Unknown (<0) or outside the costmap: defer to the map check.
        ok[inside] = values < self.cfg.costmap_reject_cost
        return ok

    def _evaluate_cluster(self, cluster, grid, field, clear_mask,
                          rx, ry, now, timing):
        cfg = self.cfg
        rows, cols = cluster.rows, cluster.cols
        cx, cy = grid.cell_to_world(cluster.centroid_row, cluster.centroid_col)
        cand = {
            'world_x': cx, 'world_y': cy, 'cell_x': 0, 'cell_y': 0,
            'size': cluster.size, 'length_m': cluster.length_m,
            'cls': fs.UNSAFE, 'reason': '', 'gain_m2': 0.0, 'gain_norm': 0.0,
            'path_m': float('nan'), 'reachable': None, 'utility': 0.0,
            'probe': False, 'probe_ready': False,
        }

        margin = int(math.ceil(cfg.map_edge_margin_m / grid.resolution))
        valid = (
            (rows >= margin) & (rows < grid.height - margin) &
            (cols >= margin) & (cols < grid.width - margin))
        if not valid.any():
            cand['reason'] = 'map_edge'
            self._edge_rejected += 1
            return cand

        safe = valid & ~clear_mask[rows, cols]
        if not safe.any():
            cand['reason'] = 'obstacle_clearance'
            self._map_safety_rejected += 1
            return cand

        xs, ys = grid.cells_to_world(rows, cols)
        usable = safe & self._costmap_ok(xs, ys)
        if not usable.any():
            cand['cls'] = fs.TEMPORARILY_BLOCKED
            cand['reason'] = 'costmap'
            self._costmap_safety_rejected += 1
            return cand

        reachable = None
        pick = usable
        if field is not None:
            reach = np.isfinite(field.distances(rows, cols))
            if (usable & reach).any():
                pick = usable & reach
                reachable = True
            else:
                reachable = False

        # An ACTUAL free frontier cell, the one closest to the centroid.
        d2 = (rows - cluster.centroid_row) ** 2 + (cols - cluster.centroid_col) ** 2
        idx = np.flatnonzero(pick)
        best = int(idx[int(np.argmin(d2[idx]))])
        row, col = int(rows[best]), int(cols[best])
        wx, wy = float(xs[best]), float(ys[best])

        euclid = math.hypot(wx - rx, wy - ry)
        direction = None
        if euclid > 1e-6:
            direction = ((wx - rx) / euclid, (wy - ry) / euclid)

        if reachable:
            path = field.distance(row, col)
            planned = field.path_direction(
                row, col, rx, ry, cfg.direction_lookahead_m)
            if planned is not None:
                direction = planned
        elif reachable is False:
            path = euclid * cfg.unreachable_distance_factor
        else:
            path = euclid

        view_bearing = math.atan2(wy - ry, wx - rx)

        t0 = time.perf_counter()
        gain_estimated = True
        try:
            gain = self._goal_gain(grid, wx, wy, view_bearing)
        except Exception as exc:
            # Clearly identified fallback: frontier length times a nominal
            # depth. Used only if ray casting itself fails.
            gain_estimated = False
            gain = cluster.length_m * min(cfg.sensor_range_m, 1.0)
            self._log_throttled(
                'gain', 10.0, 'warn', f'gain estimation failed: {exc}')
        timing['gain_ms'] += (time.perf_counter() - t0) * 1000.0

        ref = reference_gain_m2(
            cfg.sensor_range_m, math.radians(cfg.sensor_fov_deg))
        gain_norm = max(0.0, min(1.0, gain / ref)) if ref > 0.0 else 0.0

        cand.update({
            'world_x': wx, 'world_y': wy, 'cell_x': col, 'cell_y': row,
            'gain_m2': gain, 'gain_norm': gain_norm,
            'gain_estimated': gain_estimated, 'path_m': path,
            'euclid_m': euclid, 'reachable': reachable,
            'view_bearing': view_bearing,
        })

        # ── memory ───────────────────────────────────────────────────────
        rec = self.memory.find(wx, wy)
        if reachable is False:
            rec = self.memory.note_unreachable(cand, grid.seq)
        else:
            self.memory.note_reachable(rec)
            rec = self.memory.find(wx, wy)
            if (rec is not None and rec.unreachable_confirmed and reachable
                    and self.memory.can_reopen(rec)):
                # The map now connects a region a probe could not reach.
                self.memory.reopen(rec)
                self._emit('REGION_REOPENED', {
                    'x': _num(rec.x), 'y': _num(rec.y),
                    'cause': 'path_found'})

        # Failure state is isolated from spatial reachability / visit memory.
        failure_rec = self.memory.find_failure(wx, wy)

        # ── classification ───────────────────────────────────────────────
        near_metric = path if reachable else euclid
        if gain < cfg.min_information_gain_m2:
            cand['cls'], cand['reason'] = fs.LOW_INFORMATION_GAIN, 'low_gain'
        elif near_metric < cfg.min_goal_distance_m:
            # The robot is already next to this frontier and its sensors
            # have not cleared it: driving there cannot help. It is not
            # selectable, but it is reported as unexplored space (so the
            # mission ends as COMPLETE_PARTIAL, not COMPLETE).
            cand['cls'], cand['reason'] = fs.EXHAUSTED, 'too_near'
        elif rec is not None and rec.exhausted:
            cand['cls'], cand['reason'] = fs.EXHAUSTED, rec.exhausted_reason
        elif failure_rec is not None and failure_rec.exhausted:
            cand['cls'], cand['reason'] = fs.EXHAUSTED, failure_rec.exhausted_reason
        elif rec is not None and rec.unreachable_confirmed:
            cand['cls'], cand['reason'] = fs.UNREACHABLE, 'probe_failed'
        elif self.memory.in_cooldown(failure_rec, now):
            cand['cls'], cand['reason'] = fs.COOLDOWN, 'failure_cooldown'
            cand['retry_in_s'] = failure_rec.cooldown_until - now
        elif reachable is False:
            confirmed = rec.unreachable_streak >= cfg.unreachable_confirm_maps
            if confirmed and not cfg.probe_unreachable_with_nav:
                rec.unreachable_confirmed = True
                rec.ref_known_cells = grid.count_known_in_window(
                    rec.x, rec.y, cfg.reopen_radius_m)
                cand['cls'], cand['reason'] = fs.UNREACHABLE, 'no_path_on_map'
            else:
                # Not written off: the navigation planner gets the last word.
                cand['cls'] = fs.TEMPORARILY_BLOCKED
                cand['reason'] = 'no_path_on_map'
                cand['probe_ready'] = confirmed
        else:
            cand['cls'], cand['reason'] = fs.ELIGIBLE, ''

        # ── utility ──────────────────────────────────────────────────────
        if cand['cls'] == fs.ELIGIBLE or cand['probe_ready']:
            cosine, c_factor = fs.continuity_factor(
                self.memory.preferred_direction, direction,
                cfg.direction_continuity_weight)
            r_factor = self.memory.recent_factor(wx, wy)
            v_factor = self.memory.visit_factor(rec)
            cand.update({
                'direction_cosine': cosine,
                'continuity': c_factor,
                'recent': r_factor,
                'visit': v_factor,
                'failures': failure_rec.failures if failure_rec is not None else 0,
                'utility': fs.utility(
                    gain_norm, path, cfg.travel_cost_lambda,
                    c_factor, r_factor, v_factor),
            })
        return cand

    def _select(self, now, wall, pose):
        cfg = self.cfg
        grid = self.grid

        if grid is None or pose is None:
            self._log_throttled(
                'no_pose', 2.0, 'warn',
                'Frontier selection postponed: no fresh map-frame robot pose')
            self.needs_selection = True
            return

        rx, ry = pose[0], pose[1]
        timing = {k: 0.0 for k in self._last_timing}
        t_start = time.perf_counter()

        self._map_safety_rejected = 0
        self._costmap_safety_rejected = 0
        self._edge_rejected = 0

        for x, y, gained in self.memory.reopen_changed_regions(grid):
            self._emit('REGION_REOPENED', {
                'x': _num(x), 'y': _num(y), 'cause': 'map_changed',
                'new_area_m2': _num(gained)})

        # 1. frontier cells
        t0 = time.perf_counter()
        mask = detect_frontier_mask(grid)
        timing['detect_ms'] = (time.perf_counter() - t0) * 1000.0

        # 2. clusters
        t0 = time.perf_counter()
        clusters, self._small_rejected = cluster_frontiers(
            grid, mask, cfg.min_frontier_length_m, cfg.max_frontier_length_m)
        timing['cluster_ms'] = (time.perf_counter() - t0) * 1000.0

        # 3. one shared reachability / path-distance field
        t0 = time.perf_counter()
        field = None
        if cfg.reachability_enabled and clusters:
            try:
                field = compute_reachability(
                    grid, rx, ry, cfg.reachability_inflation_m,
                    cfg.reachability_resolution_m,
                    cfg.reachability_start_search_m)
            except Exception as exc:
                self._log_throttled(
                    'reach', 10.0, 'warn', f'reachability failed: {exc}')
            if field is None:
                self._log_throttled(
                    'reach_none', 10.0, 'warn',
                    'Path-distance field unavailable for this decision; '
                    'using straight-line distance')
        clear_mask = grid.obstacle_mask(cfg.goal_clearance_m)
        timing['reach_ms'] = (time.perf_counter() - t0) * 1000.0

        # 4. candidates
        t0 = time.perf_counter()
        candidates = [
            self._evaluate_cluster(
                cl, grid, field, clear_mask, rx, ry, now, timing)
            for cl in clusters
        ]
        timing['score_ms'] = max(
            0.0, (time.perf_counter() - t0) * 1000.0 - timing['gain_ms'])

        counts = {c: 0 for c in fs.ALL_CLASSES}
        for cand in candidates:
            counts[cand['cls']] += 1

        eligible = [c for c in candidates if c['cls'] == fs.ELIGIBLE]
        best = None
        if eligible:
            best = max(eligible, key=lambda c: c['utility'])
        else:
            probes = [c for c in candidates if c['probe_ready']]
            if probes:
                best = max(probes, key=lambda c: c['utility'])
                best['probe'] = True

        timing['decision_ms'] = (time.perf_counter() - t_start) * 1000.0

        self.frontier_count = len(clusters)
        self.frontier_mode = 'full'
        self.frontier_full += 1
        self.frontier_ms = timing['detect_ms'] + timing['cluster_ms']
        self._last_counts = counts
        self._last_timing = timing

        ranked = sorted(candidates, key=lambda c: c['utility'], reverse=True)
        self._emit('FRONTIER_EVALUATION', {
            'map_seq': grid.seq,
            'clusters': len(clusters),
            'rejected_small': self._small_rejected,
            'counts': counts,
            'timing_ms': {k: _num(v, 3) for k, v in timing.items()},
            'path_field': field is not None,
            'known_area_m2': _num(self.progress.known_area_m2, 3),
            'candidates': [
                {'x': _num(c['world_x'], 3), 'y': _num(c['world_y'], 3),
                 'cls': c['cls'], 'reason': c['reason'],
                 'gain_m2': _num(c['gain_m2'], 3),
                 'path_m': _num(c['path_m'], 3),
                 'utility': _num(c['utility'], 6)}
                for c in ranked[:max(0, cfg.event_max_candidates)]
            ],
        })
        if cfg.log_decisions:
            self.p.log(
                'info',
                'Frontiers: clusters=%d eligible=%d cooldown=%d blocked=%d '
                'unreachable=%d exhausted=%d low_gain=%d unsafe=%d '
                '(decision %.1f ms)' % (
                    len(clusters), counts[fs.ELIGIBLE], counts[fs.COOLDOWN],
                    counts[fs.TEMPORARILY_BLOCKED], counts[fs.UNREACHABLE],
                    counts[fs.EXHAUSTED], counts[fs.LOW_INFORMATION_GAIN],
                    counts[fs.UNSAFE], timing['decision_ms']))

        self.p.publish_markers([
            {'x': c['world_x'], 'y': c['world_y'], 'cls': c['cls'],
             'best': c is best}
            for c in candidates
        ])

        if best is not None:
            self.completion.note_activity(self._run_time)

        verdict = self.completion.evaluate(
            counts, grid.seq, self._run_time,
            self.progress.growth_over_window(now),
            selectable=(best is not None))

        if best is None:
            if verdict is not None:
                self._end(verdict['class'], verdict['reason'])
                return
            if counts[fs.COOLDOWN] > 0:
                self._idle_reason = 'WAITING_FRONTIER_RETRY'
                waits = [c.get('retry_in_s', 0.0) for c in candidates
                         if c['cls'] == fs.COOLDOWN]
                self._log_throttled(
                    'retry', 5.0, 'info',
                    f'No frontier selectable now: {counts[fs.COOLDOWN]} '
                    f'cooling down, next retry in ~{min(waits):.0f} s')
            else:
                self._idle_reason = ''
            return

        self._idle_reason = ''
        self._dispatch(best, grid, pose, now, wall)

    def _dispatch(self, best, grid, pose, now, wall):
        cfg = self.cfg
        goal = dict(best)
        goal['map_seq'] = grid.seq
        goal['known_before'] = grid.count_known_in_window(
            goal['world_x'], goal['world_y'], cfg.productivity_radius_m)
        goal['selection_robot_x'] = float(pose[0])
        goal['selection_robot_y'] = float(pose[1])

        if cfg.log_decisions:
            cosine = goal.get('direction_cosine')
            self.p.log(
                'info',
                'Target (%.2f, %.2f)%s gain=%.2f m2 path=%.2f m '
                'C=%.2f (cos=%s) R=%.2f V=%.2f U=%.5f failures=%d' % (
                    goal['world_x'], goal['world_y'],
                    ' [probe]' if goal['probe'] else '',
                    goal['gain_m2'], goal['path_m'],
                    goal.get('continuity', 1.0),
                    'na' if cosine is None else '%+.2f' % cosine,
                    goal.get('recent', 1.0), goal.get('visit', 1.0),
                    goal['utility'], goal.get('failures', 0)))

        self.p.point_sensor(goal['view_bearing'])

        # Emit before begin(): begin() can finish a transaction immediately
        # (e.g. the robot is already at the frontier), and the selection
        # event must precede its outcome in the event stream.
        txn_id = self.supervisor.next_txn_id
        self._emit('GOAL_SELECTED', {
            'x': _num(goal['world_x'], 3), 'y': _num(goal['world_y'], 3),
            'utility': _num(goal['utility'], 6),
            'gain_m2': _num(goal['gain_m2'], 3),
            'gain_norm': _num(goal['gain_norm'], 4),
            'gain_estimated': goal.get('gain_estimated', True),
            'path_m': _num(goal['path_m'], 3),
            'euclid_m': _num(goal.get('euclid_m'), 3),
            'reachable': goal['reachable'],
            'probe': goal['probe'],
            'continuity': _num(goal.get('continuity', 1.0)),
            'direction_cosine': _num(goal.get('direction_cosine')),
            'recent': _num(goal.get('recent', 1.0)),
            'visit': _num(goal.get('visit', 1.0)),
            'failures': goal.get('failures', 0),
            'frontier_cells': goal['size'],
            'decision_ms': _num(self._last_timing['decision_ms'], 3),
            'known_area_m2': _num(self.progress.known_area_m2, 3),
        }, txn_id)
        self.supervisor.begin(goal, now, wall)

    # =====================================================================
    # transaction outcome
    # =====================================================================
    def _on_transaction_finished(self, txn, outcome):
        now = self.clock.t
        goal = txn.goal
        kind = outcome['kind']
        self.completion.note_activity(self._run_time)
        self.memory.reset_travel_anchor()

        data = {
            'x': _num(goal['world_x'], 3), 'y': _num(goal['world_y'], 3),
            'outcome': kind, 'reason': outcome['reason'],
            'detail': outcome['detail'], 'action_status': outcome['status'],
            'probe': goal.get('probe', False),
            'preview_s': _num(outcome['preview_s'], 3),
            'rotation_s': _num(outcome['rotation_s'], 3),
            'nav_s': _num(outcome['nav_s'], 3),
            'budget_s': _num(outcome['budget_s'], 2),
            'budget_extensions': outcome['budget_extensions'],
            'recoveries': outcome['recoveries'],
            'distance_remaining': _num(outcome['distance_remaining'], 3),
            'utility': _num(goal.get('utility'), 6),
            'gain_m2': _num(goal.get('gain_m2'), 3),
            'path_m': _num(goal.get('path_m'), 3),
            'known_area_m2': _num(self.progress.known_area_m2, 3),
        }

        if kind in (ns.REACHED, ns.RESOLVED):
            self.memory.record_reached(goal['world_x'], goal['world_y'])
            if kind == ns.REACHED:
                self.goals_reached += 1
                self._emit('FRONTIER_REACHED', data, txn.id)
                self.p.log('info', f'Goal reached ({outcome["reason"]})')
            else:
                self.goals_resolved += 1
                data['classification'] = FRONTIER_RESOLVED
                self._emit('FRONTIER_RESOLVED', data, txn.id)
                self.p.log(
                    'info', 'Frontier already observed: goal canceled early')
            self._begin_settle(goal, kind, txn.id, now)

        elif kind == ns.FAILED:
            self.goals_failed += 1
            known = 0
            if self.grid is not None:
                known = self.grid.count_known_in_window(
                    goal['world_x'], goal['world_y'], self.cfg.reopen_radius_m)
            info = self.memory.record_failure(
                goal, now, known, probe=bool(goal.get('probe')))
            data['classification'] = NAVIGATION_FAILURE
            data['memory'] = {
                'action': info['action'], 'failures': info['failures'],
                'cooldown_s': _num(info.get('cooldown_s'), 1)}
            self._emit('NAVIGATION_FAILED', data, txn.id)
            self.p.log(
                'warn',
                f'Navigation failed ({outcome["reason"]}) near '
                f'({info["x"]:.2f}, {info["y"]:.2f}): failure '
                f'{info["failures"]} -> {info["action"]}')

        else:
            self.goals_canceled += 1
            data['classification'] = INTENTIONAL_CANCELLATION
            self._emit('NAVIGATION_CANCELED', data, txn.id)
            self.p.log('info', f'Goal canceled ({outcome["reason"]})')

        if self._motion_allowed():
            self.needs_selection = True

    # ── productive vs unproductive visits ────────────────────────────────
    def _begin_settle(self, goal, kind, txn_id, now):
        self._settle = {
            'goal': goal, 'kind': kind, 'txn_id': txn_id,
            'seq': self._map_seq,
            'deadline': now + self.cfg.settle_timeout_s,
        }
        if self.cfg.settle_maps <= 0 or self.cfg.settle_timeout_s <= 0.0:
            self._evaluate_visit(now)

    def _process_settle(self, now):
        settle = self._settle
        if settle is None:
            return
        new_maps = self._map_seq - settle['seq']
        if new_maps >= self.cfg.settle_maps or now >= settle['deadline']:
            self._evaluate_visit(now)

    def _evaluate_visit(self, now):
        """Did reaching this frontier add map near it?"""
        settle = self._settle
        self._settle = None
        goal = settle['goal']
        grid = self.grid
        cfg = self.cfg

        growth = 0.0
        after = goal.get('known_before', 0)
        if grid is not None:
            after = grid.count_known_in_window(
                goal['world_x'], goal['world_y'], cfg.productivity_radius_m)
            growth = (
                (after - goal.get('known_before', after)) *
                grid.resolution * grid.resolution)

        maps_since_arrival = self._map_seq - settle['seq']
        info = None
        if growth >= cfg.min_productive_gain_m2:
            classification = PRODUCTIVE_SUCCESS
            self.visits_productive += 1
            self.memory.record_productive(goal)
        elif maps_since_arrival <= 0:
            # No map arrived after the visit: no evidence either way.
            classification = UNVERIFIED_SUCCESS
        else:
            classification = UNPRODUCTIVE_SUCCESS
            self.visits_unproductive += 1
            known = after
            if grid is not None:
                known = grid.count_known_in_window(
                    goal['world_x'], goal['world_y'], cfg.reopen_radius_m)
            info = self.memory.record_unproductive(goal, known)

        data = {
            'x': _num(goal['world_x'], 3), 'y': _num(goal['world_y'], 3),
            'classification': classification,
            'arrival': settle['kind'],
            'new_area_m2': _num(growth, 3),
            'expected_gain_m2': _num(goal.get('gain_m2'), 3),
            'maps_since_arrival': maps_since_arrival,
            'memory': info and {
                'action': info['action'],
                'unproductive': info['unproductive']},
        }
        self._emit('VISIT_EVALUATED', data, settle['txn_id'])
        if classification == UNPRODUCTIVE_SUCCESS:
            self._emit('UNPRODUCTIVE_VISIT', data, settle['txn_id'])
            self.p.log(
                'info',
                f'Unproductive visit ({growth:.2f} m2 new) -> '
                f'{info["action"]}')

        if self._motion_allowed():
            self.needs_selection = True

    # =====================================================================
    # end of exploration
    # =====================================================================
    def _summary(self):
        return {
            'class': self.end_class,
            'reason': self.completion_reason,
            'frontier_counts': dict(self._last_counts),
            'navigation': {
                'reached': self.goals_reached,
                'failed': self.goals_failed,
                'resolved_early': self.goals_resolved,
                'canceled': self.goals_canceled,
                'productive_visits': self.visits_productive,
                'unproductive_visits': self.visits_unproductive,
                'stale_callbacks': self.supervisor.stale_callbacks,
                'preview_time_s': _num(self.supervisor.total_preview_s, 2),
                'rotation_time_s': _num(self.supervisor.total_rotation_s, 2),
                'nav_time_s': _num(self.supervisor.total_nav_s, 2),
            },
            'memory': {
                'regions': len(self.memory.records),
                'exhausted': self.memory.count_exhausted(),
                'unreachable': self.memory.count_unreachable(),
            },
            'map': {
                'known_area_m2': _num(self.progress.known_area_m2, 3),
                'growth_window_m2': _num(
                    self.progress.growth_over_window(self.clock.t), 3),
                'maps_received': self.progress.maps_received,
                'last_change_pct': _num(self.progress.last_change_pct, 5),
            },
            'run_time_s': _num(self._run_time, 2),
            'dist_travelled_m': _num(self.dist_travelled_m, 3),
            'clock': {
                'backward_jumps': self.clock.backward_jumps,
                'forward_clamps': self.clock.forward_clamps,
            },
        }

    def _end(self, end_class, reason):
        if self.ended:
            return
        self.ended = True
        self.end_class = str(end_class)
        self.completion_reason = str(reason)
        self.needs_selection = False
        self._settle = None

        if end_class == comp.COMPLETE:
            event = 'EXPLORATION_COMPLETE'
        elif end_class == comp.COMPLETE_PARTIAL:
            event = 'EXPLORATION_PARTIAL'
        elif end_class == comp.STAGNATED:
            event = 'EXPLORATION_STAGNATED'
        elif end_class == comp.TIME_LIMIT:
            event = 'EXPLORATION_TIME_LIMIT'
        elif end_class.startswith('ERROR'):
            event = 'EXPLORATION_ERROR'
        else:
            event = 'EXPLORATION_TERMINATED'
        self._emit(event, self._summary())

        self.p.log('info', '=' * 60)
        self.p.log('info', f'EXPLORATION ENDED: {end_class}')
        self.p.log('info', f'Reason: {reason}')
        self.p.log(
            'info',
            f'reached={self.goals_reached} failed={self.goals_failed} '
            f'resolved={self.goals_resolved} '
            f'unproductive={self.visits_unproductive} '
            f'known_area={self.progress.known_area_m2:.1f} m2')
        self.p.log('info', '=' * 60)

        self.p.publish_status(self.status_text())
        if self.cfg.request_map_save_on_complete:
            self._save_map()
        else:
            self.p.log('info', 'Automatic map save disabled')

    def _save_map(self):
        """Ask the program manager to save the map; never write files here."""
        self.p.publish_map_save_status(
            f'Map save requested ({self.cfg.map_save_topic})...')
        self.p.request_map_save()

    # =====================================================================
    # status + stats
    # =====================================================================
    def status_text(self):
        if self.ended:
            if self.end_class.startswith('ERROR'):
                return self.end_class
            if self.cfg.legacy_complete_status:
                return 'COMPLETE'
            return self.end_class
        if self.ctrl_state == STATE_WAITING:
            return 'WAITING_FOR_START'
        if self.ctrl_state == STATE_STOPPED:
            return 'STOPPED'
        phase = self.supervisor.phase
        if phase == ns.FAULT:
            return 'ERROR_NAV_CANCEL_UNCONFIRMED'
        if self.ctrl_state == STATE_PAUSED:
            return 'CANCELING' if phase == ns.CANCELING else 'PAUSED'
        if self.grid is None or not self._started:
            return 'WAITING_FOR_MAP'
        if phase == ns.PREVIEW:
            return 'TARGET_PREVIEW'
        if phase == ns.ROTATING:
            return 'ROTATING'
        if phase in (ns.SENDING, ns.NAVIGATING):
            return 'EXPLORING'
        if phase == ns.CANCELING:
            return 'CANCELING'
        if self._settle is not None:
            return 'SETTLING'
        if self._idle_reason:
            return self._idle_reason
        return 'IDLE'

    def stats_text(self):
        """Flat key=value CSV.

        The first block reproduces every field of the original framework in
        the same order and format; new fields are appended after it.
        """
        now = self.clock.t
        nan = float('nan')

        cooling = self.memory.active_cooldowns(now)
        bl_str = ';'.join(
            f'{rec.cell_x},{rec.cell_y},{rec.cooldown_until - now:.0f}'
            for rec in cooling)
        temp_count = len(cooling)
        soonest = (
            min(rec.cooldown_until - now for rec in cooling)
            if cooling else -1.0)

        change = self.progress.last_change_pct
        map_change_text = 'na' if change is None else f'{change:.5f}'
        frontier_change = 0.0 if change is None else float(change)

        map_width = map_height = 0
        map_res = extent_x = extent_y = 0.0
        if self.grid is not None:
            map_width = self.grid.width
            map_height = self.grid.height
            map_res = self.grid.resolution
            extent_x = map_width * map_res
            extent_y = map_height * map_res

        map_age = (
            max(0.0, now - self.progress.last_update_t)
            if self.progress.last_update_t is not None else -1.0)

        robot_x = robot_y = robot_yaw_deg = nan
        pose = self._last_pose
        if pose is not None:
            robot_x, robot_y = pose[0], pose[1]
            robot_yaw_deg = math.degrees(pose[2])

        target_x = target_y = dist_target = nan
        target_utility = target_gain = target_path = nan
        txn = self.supervisor.txn
        if txn is not None:
            goal = txn.goal
            target_x, target_y = goal['world_x'], goal['world_y']
            target_utility = goal.get('utility', nan)
            target_gain = goal.get('gain_m2', nan)
            target_path = goal.get('path_m', nan)
            if pose is not None:
                dist_target = math.hypot(target_x - robot_x, target_y - robot_y)

        counts = self._last_counts
        timing = self._last_timing
        reason = self.completion_reason.replace(',', ';').replace('=', ':')

        legacy = (
            f'node4_mode=full,'
            f'completion_reason={reason},'
            f'reached={self.goals_reached},'
            f'failed={self.goals_failed},'
            f'blacklist={bl_str},'
            f'blacklist_count={temp_count},'
            f'blacklist_soonest_s={soonest:.2f},'
            f'temp_blacklisted={temp_count},'
            f'exhausted={self.memory.count_exhausted()},'
            f'no_eligible_refreshes={self.completion.no_candidate_maps},'
            f'stable_refreshes={self.progress.stable_maps},'
            f'map_change_pct={map_change_text},'
            f'map_safety_rejected={self._map_safety_rejected},'
            f'costmap_safety_rejected={self._costmap_safety_rejected},'
            f'recent_goals={len(self.memory.recent_goals)},'
            f'frontier_count={self.frontier_count},'
            f'frontier_mode={self.frontier_mode},'
            f'frontier_ms={self.frontier_ms:.3f},'
            f'frontier_change_pct={frontier_change:.5f},'
            f'frontier_full={self.frontier_full},'
            f'frontier_incr=0,'
            f'map_width={map_width},'
            f'map_height={map_height},'
            f'map_res={map_res:.6f},'
            f'map_extent_x_m={extent_x:.3f},'
            f'map_extent_y_m={extent_y:.3f},'
            f'map_known_area_m2={self.progress.known_area_m2:.4f},'
            f'map_growth_m2_min={self.progress.growth_m2_min:.4f},'
            f'map_metric_age_s={map_age:.3f},'
            f'robot_x={robot_x:.4f},'
            f'robot_y={robot_y:.4f},'
            f'robot_yaw_deg={robot_yaw_deg:.3f},'
            f'target_x={target_x:.4f},'
            f'target_y={target_y:.4f},'
            f'dist_target_m={dist_target:.4f},'
            f'dist_travelled_m={self.dist_travelled_m:.4f}'
        )

        added = (
            f',framework_version={FRAMEWORK_VERSION},'
            f'ctrl_state={self.ctrl_state},'
            f'phase={self.supervisor.phase},'
            f'txn_id={self.supervisor.txn_id},'
            f'completion_class={self.end_class},'
            f'eligible={counts[fs.ELIGIBLE]},'
            f'cooldown={counts[fs.COOLDOWN]},'
            f'temporarily_blocked={counts[fs.TEMPORARILY_BLOCKED]},'
            f'unreachable={counts[fs.UNREACHABLE]},'
            f'exhausted_candidates={counts[fs.EXHAUSTED]},'
            f'low_gain={counts[fs.LOW_INFORMATION_GAIN]},'
            f'unsafe={counts[fs.UNSAFE]},'
            f'edge_rejected={self._edge_rejected},'
            f'small_rejected={self._small_rejected},'
            f'resolved={self.goals_resolved},'
            f'canceled={self.goals_canceled},'
            f'productive_visits={self.visits_productive},'
            f'unproductive_visits={self.visits_unproductive},'
            f'target_utility={target_utility:.6f},'
            f'target_gain_m2={target_gain:.4f},'
            f'target_path_m={target_path:.4f},'
            f'detect_ms={timing["detect_ms"]:.3f},'
            f'cluster_ms={timing["cluster_ms"]:.3f},'
            f'reach_ms={timing["reach_ms"]:.3f},'
            f'gain_ms={timing["gain_ms"]:.3f},'
            f'score_ms={timing["score_ms"]:.3f},'
            f'decision_ms={timing["decision_ms"]:.3f},'
            f'preview_time_s={self.supervisor.total_preview_s:.2f},'
            f'rotation_time_s={self.supervisor.total_rotation_s:.2f},'
            f'nav_time_s={self.supervisor.total_nav_s:.2f},'
            f'run_time_s={self._run_time:.2f},'
            f'map_growth_window_m2={self.progress.growth_over_window(now):.4f},'
            f'nav_fault={1 if self.supervisor.phase == ns.FAULT else 0},'
            f'clock_frozen={1 if self._frozen else 0},'
            f'clock_backward_jumps={self.clock.backward_jumps},'
            f'stale_callbacks={self.supervisor.stale_callbacks}'
        )
        return legacy + added
