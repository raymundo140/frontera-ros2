"""
navigation_supervisor.py — executes ONE frontier goal safely.

A goal is a transaction with a unique integer ID:

    PREVIEW -> ROTATING -> SENDING -> NAVIGATING -> (CANCELING) -> IDLE
                                                         |
                                                       FAULT

Safety properties enforced here:

* Every asynchronous navigation callback carries the transaction ID it was
  created for. A callback that belongs to an older transaction can never
  clear, fail or restart the current one. If a stale callback reports that
  an old goal was ACCEPTED, that orphan goal is canceled.
* Cancellation is a state, not an assumption. After a cancel request the
  supervisor stays in CANCELING until the navigation stack reports a
  terminal result. While it is unresolved no new goal can start.
* An unanswered cancel is retried, then escalated to FAULT. FAULT is never
  cleared by a timer, a transient action-server disconnect, or a manual
  acknowledgement: only a confirmed terminal result may release its goal.
  Recovery without a result requires the process supervisor (for example a
  program manager) to stop and restart the navigation stack and explorer
  together, not dispatch a new goal.
* Every phase has a deadline in behavior time (simulation-safe), and the
  phases that depend on an external component also have a wall-clock bound.

The supervisor does not plan, avoid obstacles or run recoveries. Those stay
with the navigation stack; this class only decides when a goal is over.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# Transaction phases (see the diagram above).
IDLE = 'IDLE'
PREVIEW = 'PREVIEW'
ROTATING = 'ROTATING'
SENDING = 'SENDING'
NAVIGATING = 'NAVIGATING'
CANCELING = 'CANCELING'
FAULT = 'FAULT'

# Raw transaction outcomes reported to the controller.
REACHED = 'REACHED'
RESOLVED = 'RESOLVED'
FAILED = 'FAILED'
CANCELED = 'CANCELED'

# Action result states (strings so this module needs no ROS import).
RESULT_SUCCEEDED = 'SUCCEEDED'
RESULT_CANCELED = 'CANCELED'
RESULT_ABORTED = 'ABORTED'

# Cancel reasons that turn a canceled/aborted result into REACHED, RESOLVED
# or FAILED; any other cancel reason yields CANCELED.
REASON_REACHED = 'reached_threshold'
REASON_RESOLVED = 'frontier_resolved'
FAILURE_REASONS = ('budget_exceeded', 'no_progress')


def _wrap(angle):
    """Wrap an angle to [-pi, pi] [rad]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class Transaction:
    """State of one goal transaction, from selection to a confirmed end.

    Times are behavior time [s] unless the name contains 'wall'.

    Attributes:
        id: Unique transaction ID; every navigation callback carries it.
        goal: Goal dictionary from the controller ('world_x', 'world_y',
            'path_m', ...).
        t_begin: Time the transaction began.
        preview_until: End of the preview phase.
        rot_target: Target heading of the rotate-first phase [rad].
        rot_start: Time the rotation began, or None if skipped.
        rot_deadline: Rotation timeout.
        rot_best: Smallest heading error reached so far [rad].
        rot_mark: Time of the last rotation progress.
        rot_pose_lost: Time the pose was lost during rotation, or None.
        handle: Navigation goal handle once accepted.
        nav_start: Time the goal was sent, or None if never sent.
        budget_s: Time budget of the goal.
        budget_deadline: Current budget deadline (moves on extension).
        extensions: Budget extensions used.
        response_deadline: Deadline for the accept/reject response.
        response_wall_deadline: Same deadline in wall time.
        progress_mark: Time of the last observed progress.
        progress_xy: Robot (x, y) at the last progress [m].
        best_remaining: Smallest distance remaining reported [m].
        last_remaining: Latest distance remaining reported [m].
        recoveries: Recovery behaviors reported by the navigation stack.
        recovery_graces_used: Recoveries that reset the progress timer.
        resolved_votes: Consecutive maps on which the frontier looked
            resolved.
        cancel_reason: Why a cancel was requested, or None.
        cancel_attempts: Cancel requests sent.
        cancel_deadline: Deadline for the cancel confirmation.
        cancel_wall_deadline: Same deadline in wall time.
        preview_s: Duration of the preview phase.
        rotation_s: Duration of the rotation phase.
        nav_s: Duration of the navigation phase.
        notes: Free-form notes for the event log.
    """

    id: int
    goal: dict
    t_begin: float
    preview_until: float = 0.0
    # rotation
    rot_target: float = 0.0
    rot_start: Optional[float] = None
    rot_deadline: float = 0.0
    rot_best: float = 0.0
    rot_mark: float = 0.0
    rot_pose_lost: Optional[float] = None
    # navigation
    handle: Any = None
    nav_start: Optional[float] = None
    budget_s: float = 0.0
    budget_deadline: float = 0.0
    extensions: int = 0
    response_deadline: float = 0.0
    response_wall_deadline: float = 0.0
    progress_mark: float = 0.0
    progress_xy: Optional[tuple] = None
    best_remaining: Optional[float] = None
    last_remaining: Optional[float] = None
    recoveries: int = 0
    recovery_graces_used: int = 0
    resolved_votes: int = 0
    # cancellation
    cancel_reason: Optional[str] = None
    cancel_attempts: int = 0
    cancel_deadline: float = 0.0
    cancel_wall_deadline: float = 0.0
    # durations (behavior time)
    preview_s: float = 0.0
    rotation_s: float = 0.0
    nav_s: float = 0.0
    notes: list = field(default_factory=list)


class NavigationSupervisor:
    """Runs one goal transaction at a time and decides when it is over.

    The controller starts a transaction with begin(), drives it with tick()
    and rotation_tick(), and forwards the navigation callbacks; the
    supervisor reports the outcome once through on_finished.

    Attributes:
        phase: Current phase (IDLE, PREVIEW, ROTATING, SENDING, NAVIGATING,
            CANCELING or FAULT).
        txn: The active Transaction, or None when IDLE.
        fault_since: Behavior time FAULT was entered [s], or None.
        fault_reason: Why FAULT was entered.
        total_preview_s: Accumulated preview time over all goals [s].
        total_rotation_s: Accumulated rotation time over all goals [s].
        total_nav_s: Accumulated navigation time over all goals [s].
        stale_callbacks: Callbacks received for finished transactions.
    """

    def __init__(self, cfg, ports, motion_allowed, on_finished, emit,
                 rotation_clear):
        """Create an idle supervisor.

        Args:
            cfg: ExplorationConfig.
            ports: Interface to the outside world (pose, navigation action,
                velocity command, goal marker, logging); provided by the
                ROS node or by a test double.
            motion_allowed: Callable returning True while exploration is
                running; checked before every motion command.
            on_finished: Callable(txn, outcome) invoked once per finished
                transaction.
            emit: Callable(event, data, txn_id) that logs a framework event.
            rotation_clear: Callable(x, y, radius_m) returning True if the
                robot can rotate in place without touching a known obstacle.
        """
        self.cfg = cfg
        self.p = ports
        self._motion_allowed = motion_allowed
        self._on_finished = on_finished
        self._emit = emit
        self._rotation_clear = rotation_clear

        self.phase = IDLE
        self.txn = None
        self._next_id = 1

        self.fault_since = None
        self.fault_reason = ''
        self._fault_server_lost = False

        self.total_preview_s = 0.0
        self.total_rotation_s = 0.0
        self.total_nav_s = 0.0
        self.stale_callbacks = 0

    @property
    def busy(self):
        """True while a transaction is active (phase other than IDLE)."""
        return self.phase != IDLE

    @property
    def txn_id(self):
        """ID of the active transaction, or 0 when IDLE."""
        return self.txn.id if self.txn is not None else 0

    @property
    def next_txn_id(self):
        """ID the next transaction will receive."""
        return self._next_id

    # =====================================================================
    # start
    # =====================================================================
    def begin(self, goal, now, wall):
        """Start a new transaction for a selected goal.

        Publishes the goal marker, then enters PREVIEW (if goal_preview_s
        is positive) or goes straight to rotation and dispatch.

        Args:
            goal: Goal dictionary; receives the key 'txn_id'.
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].

        Returns:
            The new transaction ID.

        Raises:
            RuntimeError: If a transaction is already active.
        """
        if self.busy:
            raise RuntimeError('begin() while a transaction is active')

        txn = Transaction(id=self._next_id, goal=goal, t_begin=float(now))
        self._next_id += 1
        goal['txn_id'] = txn.id
        self.txn = txn

        pose = self.p.get_pose()
        yaw = pose[2] if pose is not None else 0.0
        self.p.publish_goal(goal['world_x'], goal['world_y'], yaw)

        if self.cfg.goal_preview_s > 0.0:
            self.phase = PREVIEW
            txn.preview_until = float(now) + self.cfg.goal_preview_s
        else:
            self._start_rotation(now, wall)
        return txn.id

    # =====================================================================
    # periodic entry points
    # =====================================================================
    def tick(self, now, wall):
        """Advance the active transaction; called at decision_rate_hz.

        Ends the preview, enforces the response deadline, supervises
        navigation progress, retries unanswered cancels and observes FAULT.

        Args:
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].
        """
        phase = self.phase
        if phase == PREVIEW:
            if not self._motion_allowed():
                self._finish(CANCELED, 'not_running', now)
            elif now >= self.txn.preview_until:
                self._start_rotation(now, wall)
        elif phase == SENDING:
            txn = self.txn
            if (now >= txn.response_deadline
                    or wall >= txn.response_wall_deadline):
                self._enter_fault('goal_response_timeout', now)
        elif phase == NAVIGATING:
            self._supervise(now, wall)
        elif phase == CANCELING:
            self._check_cancel(now, wall)
        elif phase == FAULT:
            self._check_fault(now)

    def rotation_tick(self, now, wall):
        """Run one step of the rotate-first controller.

        Called at rotation_control_rate_hz. Commands a proportional angular
        velocity toward the target heading, bounded by the configured
        minimum and maximum, and ends the rotation when it is done, times
        out, stalls, or the pose is lost for too long; the goal is then
        sent to the navigation stack in every case.

        Args:
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].
        """
        if self.phase != ROTATING:
            return
        txn = self.txn
        cfg = self.cfg

        if not self._motion_allowed():
            self._stop_rotation(now)
            self._finish(CANCELED, 'not_running', now)
            return

        pose = self.p.get_pose()
        if pose is None:
            # Missing or stale TF: never rotate blind.
            self.p.cmd_rotate(0.0)
            if txn.rot_pose_lost is None:
                txn.rot_pose_lost = float(now)
            elif now - txn.rot_pose_lost > cfg.rotation_tf_loss_timeout_s:
                self._end_rotation(now, wall, 'pose_lost')
            return
        txn.rot_pose_lost = None

        diff = _wrap(txn.rot_target - pose[2])

        if abs(diff) < cfg.rotation_done_tolerance_rad:
            self._end_rotation(now, wall, None)
            return
        if now >= txn.rot_deadline:
            self._end_rotation(now, wall, 'timeout')
            return

        if abs(diff) <= txn.rot_best - cfg.rotation_min_progress_rad:
            txn.rot_best = abs(diff)
            txn.rot_mark = float(now)
        elif now - txn.rot_mark > cfg.rotation_stall_timeout_s:
            self._end_rotation(now, wall, 'stalled')
            return

        wz = cfg.rotation_kp * diff
        wz = max(-cfg.rotation_max_angular_vel,
                 min(cfg.rotation_max_angular_vel, wz))
        if abs(wz) < cfg.rotation_min_angular_vel:
            wz = math.copysign(cfg.rotation_min_angular_vel, diff)
        self.p.cmd_rotate(wz)

    def halt_direct_motion(self):
        """Stop the direct rotation command immediately.

        Used when the ROS clock freezes, since no deadline can then expire.
        """
        if self.phase == ROTATING:
            self.p.cmd_rotate(0.0)

    # =====================================================================
    # rotate-first
    # =====================================================================
    def _start_rotation(self, now, wall):
        """End the preview and begin rotate-first, or skip it and send.

        Rotation is skipped when disabled, when no pose is available, when
        the robot already faces the frontier, or when there is not enough
        clearance to turn in place.
        """
        txn = self.txn
        cfg = self.cfg
        txn.preview_s = float(now) - txn.t_begin

        if not self._motion_allowed():
            self._finish(CANCELED, 'not_running', now)
            return
        if not cfg.rotate_first_enabled:
            self._send(now, wall)
            return

        pose = self.p.get_pose()
        if pose is None:
            self._emit('ROTATION_SKIPPED', {'reason': 'no_pose'}, txn.id)
            self._send(now, wall)
            return

        rx, ry, yaw = pose
        bearing = math.atan2(
            txn.goal['world_y'] - ry, txn.goal['world_x'] - rx)
        target = _wrap(bearing + math.radians(cfg.rotation_heading_offset_deg))
        diff = _wrap(target - yaw)

        if abs(diff) < cfg.rotation_start_tolerance_rad:
            self._send(now, wall)
            return

        if (cfg.rotation_required_clearance_m > 0.0 and
                not self._rotation_clear(
                    rx, ry, cfg.rotation_required_clearance_m)):
            self._emit(
                'ROTATION_SKIPPED', {'reason': 'insufficient_clearance'},
                txn.id)
            self._send(now, wall)
            return

        txn.rot_target = target
        txn.rot_start = float(now)
        txn.rot_deadline = float(now) + cfg.rotation_timeout_s
        txn.rot_best = abs(diff)
        txn.rot_mark = float(now)
        txn.rot_pose_lost = None
        self.phase = ROTATING

    def _stop_rotation(self, now):
        """Command zero angular velocity and record the rotation time."""
        self.p.cmd_rotate(0.0)
        txn = self.txn
        if txn is not None and txn.rot_start is not None:
            txn.rotation_s = float(now) - txn.rot_start

    def _end_rotation(self, now, wall, abort_reason):
        """Finish rotate-first (logging any abort reason) and send the goal."""
        self._stop_rotation(now)
        if abort_reason is not None:
            self._emit(
                'ROTATION_ABORTED',
                {'reason': abort_reason,
                 'rotation_s': round(self.txn.rotation_s, 3)},
                self.txn.id)
        self._send(now, wall)

    # =====================================================================
    # dispatch
    # =====================================================================
    def _budget(self, goal):
        """Return the time budget of a goal [s].

        budget = nav_budget_base_s + nav_budget_factor * path / speed,
        clamped to [nav_budget_min_s, nav_budget_max_s], where path is the
        wavefront path length to the goal [m] and speed is
        expected_speed_mps.
        """
        cfg = self.cfg
        path = max(0.0, float(goal.get('path_m', 0.0)))
        budget = (
            cfg.nav_budget_base_s +
            cfg.nav_budget_factor * path / cfg.expected_speed_mps)
        return max(cfg.nav_budget_min_s, min(cfg.nav_budget_max_s, budget))

    def _send(self, now, wall):
        """Send the goal to the navigation stack and enter SENDING.

        Finishes the transaction instead when exploration is no longer
        running, when the robot is already within goal_reached_distance_m,
        or when the navigation server is unavailable.
        """
        txn = self.txn
        cfg = self.cfg
        goal = txn.goal

        # HARD SAFETY GATE: no navigation command unless exploration is
        # running, whatever path led here.
        if not self._motion_allowed():
            self._finish(CANCELED, 'not_running', now)
            return

        pose = self.p.get_pose()
        if pose is not None:
            dist = math.hypot(
                goal['world_x'] - pose[0], goal['world_y'] - pose[1])
            if dist < cfg.goal_reached_distance_m:
                self._finish(REACHED, 'already_within_threshold', now)
                return

        if not self.p.nav_ready():
            self._finish(CANCELED, 'nav_unavailable', now)
            return

        yaw = pose[2] if pose is not None else 0.0

        txn.nav_start = float(now)
        txn.budget_s = self._budget(goal)
        txn.budget_deadline = float(now) + txn.budget_s
        txn.progress_mark = float(now)
        txn.progress_xy = (pose[0], pose[1]) if pose is not None else None
        txn.response_deadline = float(now) + cfg.goal_response_timeout_s
        txn.response_wall_deadline = (
            float(wall) + cfg.goal_response_wall_timeout_s)

        self.phase = SENDING
        sent = self.p.nav_send(txn.id, goal['world_x'], goal['world_y'], yaw)
        if not sent:
            if self.txn is txn and self.phase == SENDING:
                self._finish(CANCELED, 'send_error', now)
            return

        self.p.publish_goal(goal['world_x'], goal['world_y'], yaw)
        self._emit(
            'NAVIGATION_STARTED',
            {'x': goal['world_x'], 'y': goal['world_y'],
             'path_m': goal.get('path_m'),
             'budget_s': round(txn.budget_s, 2),
             'preview_s': round(txn.preview_s, 3),
             'rotation_s': round(txn.rotation_s, 3)},
            txn.id)

    # =====================================================================
    # supervision while navigating
    # =====================================================================
    def _mark_progress(self, now):
        """Reset the no-progress timer."""
        self.txn.progress_mark = float(now)

    def _supervise(self, now, wall):
        """Check an active navigation goal for arrival, budget and progress.

        The goal is canceled as reached within goal_reached_distance_m.
        When the budget expires, it is extended once by half if the robot
        progressed recently, and canceled otherwise. Without
        progress_min_improvement_m of progress for progress_timeout_s, it
        is canceled as no_progress.
        """
        txn = self.txn
        cfg = self.cfg
        goal = txn.goal

        pose = self.p.get_pose()
        if pose is not None:
            dist = math.hypot(
                goal['world_x'] - pose[0], goal['world_y'] - pose[1])
            if dist < cfg.goal_reached_distance_m:
                self.request_cancel(REASON_REACHED, now, wall)
                return

            # Progress = real displacement of the robot.
            if txn.progress_xy is None:
                txn.progress_xy = (pose[0], pose[1])
            elif (math.hypot(pose[0] - txn.progress_xy[0],
                             pose[1] - txn.progress_xy[1])
                    >= cfg.progress_min_improvement_m):
                txn.progress_xy = (pose[0], pose[1])
                self._mark_progress(now)

        if now >= txn.budget_deadline:
            recent_progress = (
                now - txn.progress_mark <= 0.5 * cfg.progress_timeout_s)
            if txn.extensions < cfg.nav_budget_extensions_max and recent_progress:
                txn.extensions += 1
                txn.budget_deadline = float(now) + 0.5 * txn.budget_s
                self._emit(
                    'NAV_BUDGET_EXTENDED',
                    {'extension_s': round(0.5 * txn.budget_s, 2),
                     'distance_remaining': txn.last_remaining},
                    txn.id)
            else:
                self.request_cancel('budget_exceeded', now, wall)
            return

        if now - txn.progress_mark > cfg.progress_timeout_s:
            self.request_cancel('no_progress', now, wall)

    def vote_resolved(self, resolved, now, wall):
        """Count one map's evidence that the target frontier is resolved.

        Called once per new map while navigating. The frontier must look
        resolved on resolved_confirm_maps CONSECUTIVE maps before the goal
        is canceled, so occupancy-grid noise cannot cancel goals.

        Args:
            resolved: True if the frontier looks resolved on this map.
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].
        """
        if self.phase != NAVIGATING:
            return
        txn = self.txn
        txn.resolved_votes = txn.resolved_votes + 1 if resolved else 0
        if txn.resolved_votes >= self.cfg.resolved_confirm_maps:
            self.request_cancel(REASON_RESOLVED, now, wall)

    # =====================================================================
    # cancellation
    # =====================================================================
    def request_cancel(self, reason, now, wall):
        """Ask the active transaction to stop.

        Before a goal exists (PREVIEW, ROTATING) the transaction ends
        immediately. A goal in flight (SENDING) is canceled as soon as it is
        accepted. An active goal (NAVIGATING) enters CANCELING and stays
        there until the navigation stack confirms a terminal result.

        Args:
            reason: Cancel reason; decides how the result is classified.
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].

        Returns:
            True if a transaction was active, False when IDLE.
        """
        phase = self.phase
        txn = self.txn
        if phase == IDLE or txn is None:
            return False

        if phase in (PREVIEW, ROTATING):
            # No navigation action exists yet: cleanup is immediate.
            if phase == ROTATING:
                self._stop_rotation(now)
            self._finish(CANCELED, reason, now)
            return True

        if phase == SENDING:
            # The goal request is in flight. It is canceled the moment the
            # navigation stack accepts it (see on_goal_response).
            if txn.cancel_reason is None:
                txn.cancel_reason = reason
            return True

        if phase == NAVIGATING:
            txn.cancel_reason = reason
            self.phase = CANCELING
            self._send_cancel(now, wall)
            return True

        # CANCELING / FAULT: a cancel is already unresolved; keep its reason.
        return True

    def _send_cancel(self, now, wall):
        """Send one cancel request and arm its confirmation deadlines."""
        txn = self.txn
        txn.cancel_attempts += 1
        txn.cancel_deadline = float(now) + self.cfg.cancel_timeout_s
        txn.cancel_wall_deadline = float(wall) + self.cfg.cancel_wall_timeout_s
        self._emit(
            'NAVIGATION_CANCEL_REQUESTED',
            {'reason': txn.cancel_reason, 'attempt': txn.cancel_attempts},
            txn.id)
        self.p.nav_cancel(txn.handle)

    def _check_cancel(self, now, wall):
        """Retry an unconfirmed cancel, then escalate to FAULT."""
        txn = self.txn
        if now < txn.cancel_deadline and wall < txn.cancel_wall_deadline:
            return
        if txn.cancel_attempts <= self.cfg.cancel_retries:
            self._send_cancel(now, wall)
        else:
            self._enter_fault('cancel_unconfirmed', now)

    # =====================================================================
    # fault
    # =====================================================================
    def _enter_fault(self, reason, now):
        """Enter FAULT: no new goal until a terminal result arrives."""
        self.phase = FAULT
        self.fault_since = float(now)
        self.fault_reason = reason
        self._fault_server_lost = False
        self._emit(
            'NAV_FAULT',
            {'reason': reason,
             'cancel_reason': self.txn.cancel_reason if self.txn else None},
            self.txn_id)
        self.p.log(
            'error',
            f'Navigation stack did not answer ({reason}). The robot may '
            'still be moving. No new goal will be sent until its terminal '
            'action result is received. If Nav2 cannot produce a result, '
            'STOP the entire stack and restart it together with this node.')

    def _check_fault(self, now):
        """Observe the server state, but never infer goal termination from it.

        A brief DDS discovery outage can make server_is_ready() false even
        while Nav2 still owns an active goal. Reappearance is NOT proof that
        the old goal has stopped. A terminal NavigateToPose result is proof.
        """
        ready = self.p.nav_ready()
        if not ready and not self._fault_server_lost:
            self._fault_server_lost = True
            self._emit('NAV_SERVER_UNAVAILABLE_DURING_FAULT', {}, self.txn_id)
        # Deliberately remain in FAULT even if ready becomes True.

    def acknowledge_fault(self, now):
        """Reject unsafe manual fault clearing.

        RESET_NAV_FAULT must not release an unconfirmed Nav2 transaction.
        A terminal action result will automatically clear FAULT via
        on_result(); otherwise stop and restart the whole stack.
        """
        if self.phase != FAULT:
            return False
        self._emit('NAV_FAULT_RESET_DENIED', {
            'reason': self.fault_reason,
            'nav_server_ready': bool(self.p.nav_ready()),
            'action_result_confirmed': False,
        }, self.txn_id)
        self.p.log(
            'warn',
            'RESET_NAV_FAULT denied: the previous Nav2 goal has no terminal '
            'result. No new goal can be sent; stop and restart the complete '
            'navigation/exploration stack if Nav2 cannot respond.')
        return False

    # =====================================================================
    # asynchronous navigation callbacks (all carry their transaction ID)
    # =====================================================================
    def _is_current(self, txn_id):
        """True if txn_id belongs to the active transaction."""
        return self.txn is not None and int(txn_id) == self.txn.id

    def on_goal_response(self, txn_id, accepted, handle, error, now, wall):
        """Handle the navigation stack's accept/reject response.

        A response for an older transaction is ignored, and an orphan goal
        it reports as accepted is canceled. A transport error enters FAULT,
        since it does not prove the goal was rejected.

        Args:
            txn_id: Transaction the response belongs to.
            accepted: True if the goal was accepted.
            handle: Goal handle, used to cancel the goal later.
            error: Exception raised while sending, or None.
            now: Current behavior time [s].
            wall: Current monotonic wall time [s].
        """
        if not self._is_current(txn_id):
            self.stale_callbacks += 1
            if accepted and handle is not None:
                # An old goal was accepted after its transaction ended.
                self.p.nav_cancel(handle)
                self._emit('ORPHAN_GOAL_CANCELED', {'stale_txn': txn_id},
                           self.txn_id)
            else:
                self._emit('STALE_CALLBACK',
                           {'kind': 'goal_response', 'stale_txn': txn_id},
                           self.txn_id)
            return

        txn = self.txn
        late = self.phase == FAULT and txn.handle is None
        if self.phase != SENDING and not late:
            return                      # duplicate response

        if error is not None:
            # The send-goal future failed, but we cannot prove the goal was
            # never accepted by the remote action server. Do not dispatch a
            # competing goal based on a transport exception.
            self._enter_fault('goal_response_transport_error', now)
            return

        if not accepted:
            if txn.cancel_reason is not None or late:
                self._finish(
                    CANCELED, txn.cancel_reason or 'goal_response_timeout',
                    now)
            else:
                self._finish(CANCELED, 'rejected', now)
            return

        txn.handle = handle

        if late:
            # Accepted after the response deadline: cancel it explicitly.
            txn.cancel_reason = txn.cancel_reason or 'goal_response_timeout'
            self.fault_since = None
            self.fault_reason = ''
            self.phase = CANCELING
            self._send_cancel(now, wall)
            return

        if txn.cancel_reason is not None:
            # PAUSE / STOP arrived while the request was in flight.
            self.phase = CANCELING
            self._send_cancel(now, wall)
            return

        self.phase = NAVIGATING
        txn.progress_mark = float(now)

    def on_feedback(self, txn_id, distance_remaining, recoveries, now):
        """Handle navigation feedback.

        A drop in the remaining distance of at least
        progress_min_improvement_m counts as progress. A new recovery
        behavior resets the progress timer, at most
        progress_recovery_graces times per goal.

        Args:
            txn_id: Transaction the feedback belongs to.
            distance_remaining: Remaining path length reported [m].
            recoveries: Number of recoveries reported so far.
            now: Current behavior time [s].
        """
        if not self._is_current(txn_id) or self.phase != NAVIGATING:
            return
        txn = self.txn

        try:
            remaining = float(distance_remaining)
        except (TypeError, ValueError):
            remaining = float('nan')

        # Some stacks report 0 before the first path exists; ignore that.
        if math.isfinite(remaining) and remaining > 0.01:
            txn.last_remaining = remaining
            if (txn.best_remaining is None or
                    remaining <= txn.best_remaining
                    - self.cfg.progress_min_improvement_m):
                txn.best_remaining = remaining
                self._mark_progress(now)

        try:
            recoveries = int(recoveries)
        except (TypeError, ValueError):
            recoveries = 0
        if recoveries > txn.recoveries:
            txn.recoveries = recoveries
            # A recovery behavior is the stack working on the problem. Give
            # it time, but only a bounded number of times per goal.
            if txn.recovery_graces_used < self.cfg.progress_recovery_graces:
                txn.recovery_graces_used += 1
                self._mark_progress(now)
            self._emit('NAV_RECOVERY', {'recoveries': recoveries}, txn.id)

    def on_result(self, txn_id, status, error, now):
        """Handle the terminal navigation result and finish the transaction.

        The raw status is combined with the cancel reason to classify the
        outcome as REACHED, RESOLVED, FAILED or CANCELED. Only a verified
        terminal status releases the transaction; a transport error or an
        unknown status enters FAULT instead.

        Args:
            txn_id: Transaction the result belongs to.
            status: RESULT_SUCCEEDED, RESULT_CANCELED or RESULT_ABORTED.
            error: Exception raised while retrieving the result, or None.
            now: Current behavior time [s].
        """
        if not self._is_current(txn_id):
            self.stale_callbacks += 1
            self._emit('STALE_CALLBACK',
                       {'kind': 'result', 'stale_txn': txn_id, 'status': status},
                       self.txn_id)
            return
        if self.phase not in (NAVIGATING, CANCELING, FAULT):
            return

        txn = self.txn
        reason = txn.cancel_reason

        if error is not None:
            # An exception retrieving the result is NOT a terminal Nav2
            # result. Hold the transaction until a verified action outcome
            # or full-stack restart.
            self._enter_fault('result_transport_error', now)
            self._emit('NAV_RESULT_UNCONFIRMED',
                       {'error': str(error)}, txn.id)
            return
        elif status == RESULT_SUCCEEDED:
            kind, why = REACHED, 'succeeded'
        elif status in (RESULT_CANCELED, RESULT_ABORTED):
            if reason == REASON_REACHED:
                kind, why = REACHED, reason
            elif reason == REASON_RESOLVED:
                kind, why = RESOLVED, reason
            elif reason in FAILURE_REASONS:
                kind, why = FAILED, reason
            elif reason is not None:
                kind, why = CANCELED, reason
            elif status == RESULT_CANCELED:
                # Canceled by someone else: not evidence about the frontier.
                kind, why = CANCELED, 'external'
            else:
                kind, why = FAILED, 'aborted'
        else:
            # A nonterminal/unknown status is not proof Nav2 stopped.
            self._enter_fault('unexpected_result_status', now)
            self._emit('NAV_RESULT_UNCONFIRMED',
                       {'status': status}, txn.id)
            return

        self._finish(kind, why, now, status=status)

    # =====================================================================
    # finish
    # =====================================================================
    def _finish(self, kind, reason, now, detail=None, status=None):
        """Close the active transaction and report its outcome.

        Args:
            kind: REACHED, RESOLVED, FAILED or CANCELED.
            reason: Reason string for the event log.
            now: Current behavior time [s].
            detail: Optional extra information.
            status: Raw navigation result status, if any.
        """
        txn = self.txn
        if txn is None:
            return

        if txn.nav_start is not None:
            txn.nav_s = max(0.0, float(now) - txn.nav_start)
        if txn.preview_s == 0.0 and txn.rot_start is None \
                and txn.nav_start is None:
            txn.preview_s = max(0.0, float(now) - txn.t_begin)

        self.total_preview_s += txn.preview_s
        self.total_rotation_s += txn.rotation_s
        self.total_nav_s += txn.nav_s

        outcome = {
            'kind': kind,
            'reason': reason,
            'detail': detail,
            'status': status,
            'preview_s': txn.preview_s,
            'rotation_s': txn.rotation_s,
            'nav_s': txn.nav_s,
            'recoveries': txn.recoveries,
            'budget_s': txn.budget_s,
            'budget_extensions': txn.extensions,
            'distance_remaining': txn.last_remaining,
        }

        self.phase = IDLE
        self.txn = None
        self.fault_since = None
        self.fault_reason = ''
        self._fault_server_lost = False

        self._on_finished(txn, outcome)
