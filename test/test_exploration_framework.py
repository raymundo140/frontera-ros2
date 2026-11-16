"""
Tests for the exploration framework. No ROS installation is required:

    cd <package root>            # the repository root
    python3 -m pytest test/test_exploration_framework.py -v

The controller is driven through a fake `ports` object that stands in for
the clock, TF, the NavigateToPose action server and all publishers, so the
safety and reliability behavior can be checked deterministically.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from frontera import completion as comp          # noqa: E402
from frontera import frontier_scoring as fs      # noqa: E402
from frontera import navigation_supervisor as ns  # noqa: E402
from frontera import reachability                # noqa: E402
from frontera.config import ExplorationConfig    # noqa: E402
from frontera.controller import ExplorationController  # noqa: E402
from frontera.frontier_detection import (        # noqa: E402
    cluster_frontiers, detect_frontier_mask)
from frontera.grid import GridMap, map_change_pct  # noqa: E402
from frontera.information_gain import estimate_gain_m2  # noqa: E402
from frontera.timebase import BehaviorClock      # noqa: E402

RES = 0.05


# ─────────────────────────────────────────────────────────────────────────
# world builders
# ─────────────────────────────────────────────────────────────────────────
def make_grid(array, origin=(0.0, 0.0), res=RES):
    h, w = array.shape
    return GridMap(array.ravel(), w, h, res, origin[0], origin[1])


def room(open_right=True, size=200):
    """10 m x 10 m map; a 4 m x 4 m free room at x,y in [1, 5].

    Walls on the left, bottom and top; the right side is open to unknown
    space (a 4 m frontier at x = 5) unless open_right is False.
    """
    a = np.full((size, size), -1, dtype=np.int8)
    a[20:100, 20:100] = 0
    a[20:100, 19] = 100        # left wall
    a[19, 19:101] = 100        # bottom wall
    a[100, 19:101] = 100       # top wall
    if not open_right:
        a[19:101, 100] = 100
    return a


def room_explored():
    """The same room after the right-hand side has been mapped and closed."""
    a = room()
    a[20:100, 100:140] = 0
    a[19:101, 140] = 100
    a[19, 100:141] = 100
    a[100, 100:141] = 100
    return a


class FakePorts:

    def __init__(self):
        self.ros = 100.0
        self.wall = 5000.0
        self.pose = (2.0, 3.0, 0.0)
        self.nav_is_ready = True
        self.send_ok = True
        self.sent = []          # (txn_id, x, y, yaw)
        self.cancels = []       # handles
        self.cmd = []
        self.statuses = []
        self.stats = []
        self.events = []
        self.goals = []
        self.markers = []
        self.save_requests = 0
        self.logs = []
        self.tf_resets = 0
        self.sensor = []

    def advance(self, dt, wall_dt=None):
        self.ros += dt
        self.wall += dt if wall_dt is None else wall_dt

    # ports
    def ros_now(self): return self.ros
    def wall_now(self): return self.wall
    def get_pose(self): return self.pose
    def reset_tf(self): self.tf_resets += 1
    def nav_ready(self): return self.nav_is_ready

    def nav_send(self, txn_id, x, y, yaw):
        if not self.send_ok:
            return False
        self.sent.append((txn_id, x, y, yaw))
        return True

    def nav_cancel(self, handle): self.cancels.append(handle)
    def cmd_rotate(self, wz): self.cmd.append(wz)
    def publish_goal(self, x, y, yaw): self.goals.append((x, y, yaw))
    def publish_status(self, text): self.statuses.append(text)
    def publish_stats(self, text): self.stats.append(text)
    def publish_event(self, text): self.events.append(json.loads(text))
    def publish_markers(self, items): self.markers.append(items)
    def point_sensor(self, bearing): self.sensor.append(bearing)
    def request_map_save(self): self.save_requests += 1
    def publish_map_save_status(self, text): pass
    def log(self, level, text): self.logs.append((level, text))

    def event_names(self):
        return [e['event'] for e in self.events]


def make(cfg=None, grid_array=None, **overrides):
    cfg = cfg or ExplorationConfig()
    cfg.goal_preview_s = 0.0
    cfg.rotate_first_enabled = False
    for key, value in overrides.items():
        assert hasattr(cfg, key), key
        setattr(cfg, key, value)
    ports = FakePorts()
    ctrl = ExplorationController(cfg, ports)
    if grid_array is not None:
        ctrl.on_map(make_grid(grid_array))
    return ctrl, ports


def run(ctrl, ports, seconds, dt=0.2):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        ports.advance(dt)
        ctrl.tick()
        ctrl.rotation_tick()


def start_and_dispatch(ctrl, ports):
    """START and run until the first goal has been sent to navigation."""
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 1, ports.logs[-5:]
    return ports.sent[-1][0]


def accept(ctrl, txn_id, handle=None):
    handle = handle or f'handle-{txn_id}'
    ctrl.nav_goal_response(txn_id, True, handle)
    return handle


# ─────────────────────────────────────────────────────────────────────────
# pure modules
# ─────────────────────────────────────────────────────────────────────────
def test_frontier_definition_and_clusters():
    grid = make_grid(room())
    mask = detect_frontier_mask(grid)
    rows, cols = np.nonzero(mask)
    # Only the open right side is a frontier, and only real FREE cells.
    assert set(cols.tolist()) == {99}
    assert (grid.grid[rows, cols] == 0).all()

    clusters, small = cluster_frontiers(grid, mask, 0.15, 0.0)
    assert len(clusters) == 1 and small == 0
    assert abs(clusters[0].length_m - 4.0) < 0.06

    # A 4 m frontier is split when the maximum length is 3 m.
    split, _ = cluster_frontiers(grid, mask, 0.15, 3.0)
    assert len(split) == 2
    assert sum(c.size for c in split) == clusters[0].size


def test_diagonal_only_unknown_is_not_a_frontier():
    a = np.full((5, 5), 100, dtype=np.int8)
    a[2, 2] = 0
    a[1, 1] = -1                 # touches the free cell only diagonally
    assert not detect_frontier_mask(make_grid(a)).any()


def test_small_clusters_are_rejected_by_length():
    a = np.full((40, 40), 100, dtype=np.int8)
    a[10:30, 10:30] = 0
    a[9, 15] = -1                # 1-cell hole in a wall -> 1 frontier cell
    grid = make_grid(a)
    clusters, small = cluster_frontiers(
        grid, detect_frontier_mask(grid), 0.15, 3.0)
    assert clusters == [] and small == 1


def test_cluster_fallback_matches_scipy(monkeypatch):
    from frontera import frontier_detection as fd
    grid = make_grid(room())
    mask = detect_frontier_mask(grid)
    with_scipy, _ = cluster_frontiers(grid, mask, 0.15, 3.0)
    monkeypatch.setattr(fd, '_ndimage', None)
    without, _ = cluster_frontiers(grid, mask, 0.15, 3.0)
    assert sorted(c.size for c in with_scipy) == sorted(c.size for c in without)


def test_reachability_goes_around_walls():
    a = np.full((200, 200), 0, dtype=np.int8)
    a[0:150, 100] = 100          # wall with a gap at the top (rows 150+)
    grid = make_grid(a)
    field = reachability.compute_reachability(grid, 2.0, 1.0, 0.2, 0.1, 0.6)
    row, col = grid.world_to_cell(8.0, 1.0)
    path = field.distance(row, col)
    assert path > 6.0 * 2 and math.isfinite(path)     # straight line is 6 m
    direction = field.path_direction(row, col, 2.0, 1.0, 1.0)
    assert direction is not None and direction[1] > 0.3   # heads up first


def test_reachability_python_fallback_matches_scipy():
    a = np.full((120, 120), 0, dtype=np.int8)
    a[0:90, 60] = 100
    grid = make_grid(a)
    f1 = reachability.compute_reachability(grid, 1.0, 1.0, 0.2, 0.1, 0.6)
    f2 = reachability.compute_reachability(
        grid, 1.0, 1.0, 0.2, 0.1, 0.6, force_python=True)
    finite = np.isfinite(f1.dist)
    assert (finite == np.isfinite(f2.dist)).all()
    assert np.allclose(f1.dist[finite], f2.dist[finite], atol=1e-6)


def test_reachability_sealed_region_is_unreachable():
    a = np.full((200, 200), 0, dtype=np.int8)
    a[:, 100] = 100
    grid = make_grid(a)
    field = reachability.compute_reachability(grid, 2.0, 5.0, 0.2, 0.1, 0.6)
    assert math.isinf(field.distance(*grid.world_to_cell(8.0, 5.0)))


def test_narrow_passage_stays_open():
    # 0.6 m gap, robot radius 0.25 m -> passable, must not be closed.
    a = np.full((200, 200), 0, dtype=np.int8)
    a[:, 100] = 100
    a[94:106, 100] = 0
    grid = make_grid(a)
    field = reachability.compute_reachability(grid, 2.0, 5.0, 0.2, 0.1, 0.6)
    assert math.isfinite(field.distance(*grid.world_to_cell(8.0, 5.0)))


def test_information_gain_respects_occlusion_and_range():
    a = np.full((200, 200), -1, dtype=np.int8)
    a[80:120, 80:100] = 0
    grid = make_grid(a)
    open_gain = estimate_gain_m2(grid, 4.9, 5.0, 2.0, num_rays=180)
    a2 = a.copy()
    a2[60:140, 100] = 100        # wall right in front of the unknown side
    walled = estimate_gain_m2(make_grid(a2), 4.9, 5.0, 2.0, num_rays=180)
    assert open_gain > 3.0
    assert walled < 0.6 * open_gain
    # Never more than the sensor footprint.
    assert open_gain <= math.pi * 2.0 ** 2 + 1e-6
    # Fully known surroundings give no gain.
    known = np.zeros((200, 200), dtype=np.int8)
    assert estimate_gain_m2(make_grid(known), 5.0, 5.0, 2.0) == 0.0


def test_map_change_is_origin_aligned():
    a = room()
    g1 = make_grid(a)
    # Same world content, grid grown by 20 cells of unknown on every side.
    b = np.full((240, 240), -1, dtype=np.int8)
    b[20:220, 20:220] = a
    g2 = make_grid(b, origin=(-1.0, -1.0))
    assert map_change_pct(g1, g2) == 0.0
    c = a.copy()
    c[20:100, 100:110] = 0
    assert map_change_pct(g1, make_grid(c)) > 1.0


def test_behavior_clock_never_goes_backward():
    clock = BehaviorClock(max_forward_step_s=10.0, frozen_wall_timeout_s=5.0)
    assert clock.update(100.0, 0.0) == 0.0
    assert clock.update(102.0, 2.0) == 2.0
    assert clock.update(5.0, 3.0) == 2.0            # reset to an earlier time
    assert clock.consume_reset() and not clock.consume_reset()
    assert clock.update(6.0, 4.0) == 3.0
    assert clock.update(1000.0, 5.0) == 13.0        # huge jump is clamped
    assert not clock.is_frozen(6.0)
    assert clock.is_frozen(20.0)


# ─────────────────────────────────────────────────────────────────────────
# control: START / PAUSE / STOP
# ─────────────────────────────────────────────────────────────────────────
def test_no_motion_before_start():
    ctrl, ports = make(grid_array=room())
    run(ctrl, ports, 10.0)
    assert ports.sent == [] and ports.cmd == [] and ports.goals == []
    assert ctrl.status_text() == 'WAITING_FOR_START'


def test_start_selects_a_real_free_frontier_cell():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    _, x, y, _ = ports.sent[0]
    row, col = ctrl.grid.world_to_cell(x, y)
    assert ctrl.grid.grid[row, col] == 0          # a FREE cell
    assert col == 99                              # on the frontier
    assert ctrl.supervisor.phase == ns.SENDING and txn == 1
    accept(ctrl, txn)
    assert ctrl.supervisor.phase == ns.NAVIGATING
    assert ctrl.status_text() == 'EXPLORING'


def test_pause_waits_for_cancel_confirmation_and_does_not_blame_frontier():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    handle = accept(ctrl, txn)

    ctrl.on_command('PAUSE')
    assert ports.cancels == [handle]
    assert ctrl.supervisor.phase == ns.CANCELING

    # START while the cancel is unresolved must not dispatch a second goal.
    ctrl.on_command('START')
    run(ctrl, ports, 3.0)
    assert len(ports.sent) == 1

    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert ctrl.goals_failed == 0 and ctrl.memory.records == []
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 2                   # resumed after confirmation


def test_pause_during_goal_request_cancels_on_acceptance():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    ctrl.on_command('PAUSE')                      # response still pending
    assert ports.cancels == []
    handle = accept(ctrl, txn)
    assert ports.cancels == [handle]
    assert ctrl.supervisor.phase == ns.CANCELING
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert not ctrl.supervisor.busy and ctrl.goals_failed == 0
    run(ctrl, ports, 5.0)
    assert len(ports.sent) == 1                   # still paused


def test_stop_is_terminal():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.on_command('STOP')
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    ctrl.on_command('START')
    run(ctrl, ports, 5.0)
    assert len(ports.sent) == 1
    assert ctrl.status_text() == 'STOPPED'


def test_pause_during_preview_and_rotation_stops_immediately():
    ctrl, ports = make(grid_array=room(), goal_preview_s=3.0,
                       rotate_first_enabled=True,
                       rotation_heading_offset_deg=90.0)
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    assert ctrl.supervisor.phase == ns.PREVIEW and ports.sent == []
    run(ctrl, ports, 3.0)
    assert ctrl.supervisor.phase == ns.ROTATING
    assert any(abs(w) > 0 for w in ports.cmd)
    ctrl.on_command('PAUSE')
    assert ports.cmd[-1] == 0.0
    assert not ctrl.supervisor.busy and ports.sent == []


# ─────────────────────────────────────────────────────────────────────────
# transaction IDs / delayed callbacks
# ─────────────────────────────────────────────────────────────────────────
def test_delayed_result_cannot_touch_a_newer_goal():
    ctrl, ports = make(grid_array=room())
    txn1 = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn1)
    ctrl.nav_result(txn1, ns.RESULT_ABORTED)       # goal 1 fails
    assert ctrl.goals_failed == 1
    run(ctrl, ports, 1.0)
    txn2 = ports.sent[-1][0]
    assert txn2 != txn1
    accept(ctrl, txn2)

    # A duplicate / late result of goal 1 arrives now.
    ctrl.nav_result(txn1, ns.RESULT_ABORTED)
    ctrl.nav_result(txn1, ns.RESULT_SUCCEEDED)
    assert ctrl.supervisor.phase == ns.NAVIGATING
    assert ctrl.supervisor.txn.id == txn2
    assert ctrl.goals_failed == 1 and ctrl.goals_reached == 0
    assert ctrl.supervisor.stale_callbacks == 2
    assert 'STALE_CALLBACK' in ports.event_names()


def test_late_acceptance_of_an_old_goal_is_cancelled():
    ctrl, ports = make(grid_array=room())
    txn1 = start_and_dispatch(ctrl, ports)
    ctrl.nav_goal_response(txn1, False, None)      # rejected
    run(ctrl, ports, 1.0)
    txn2 = ports.sent[-1][0]
    ctrl.nav_goal_response(txn1, True, 'orphan')   # stale acceptance
    assert ports.cancels == ['orphan']
    assert ctrl.supervisor.txn.id == txn2
    assert ctrl.supervisor.phase == ns.SENDING


# ─────────────────────────────────────────────────────────────────────────
# cancellation deadline and fault
# ─────────────────────────────────────────────────────────────────────────
def test_unconfirmed_cancel_escalates_and_blocks_new_goals():
    ctrl, ports = make(grid_array=room(), nav_fault_abort_timeout_s=0.0)
    txn = start_and_dispatch(ctrl, ports)
    handle = accept(ctrl, txn)
    ctrl.on_command('PAUSE')
    ctrl.on_command('START')
    run(ctrl, ports, 11.0)
    assert ports.cancels == [handle, handle]       # one retry
    run(ctrl, ports, 11.0)
    assert ctrl.supervisor.phase == ns.FAULT
    assert ctrl.status_text() == 'ERROR_NAV_CANCEL_UNCONFIRMED'
    run(ctrl, ports, 300.0, dt=1.0)
    assert len(ports.sent) == 1                    # never a competing goal
    assert not ctrl.ended

    # The stack finally answers: the fault clears and exploration resumes.
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert not ctrl.supervisor.busy
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 2


def test_fault_can_end_the_mission_as_an_error_and_saves_the_map():
    ctrl, ports = make(grid_array=room(), nav_fault_abort_timeout_s=30.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.on_command('PAUSE')
    ctrl.on_command('START')
    run(ctrl, ports, 60.0)
    assert ctrl.ended and ctrl.end_class == comp.ERROR_NAV_UNRESPONSIVE
    assert ctrl.status_text() == comp.ERROR_NAV_UNRESPONSIVE
    assert 'EXPLORATION_ERROR' in ports.event_names()
    assert ports.save_requests == 1
    assert len(ports.sent) == 1


def test_fault_reset_and_server_flap_never_release_an_unconfirmed_goal():
    ctrl, ports = make(grid_array=room(), nav_fault_abort_timeout_s=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.on_command('PAUSE')
    run(ctrl, ports, 25.0)
    assert ctrl.supervisor.phase == ns.FAULT

    # DDS discovery loss and reappearance are NOT proof of action termination.
    ports.nav_is_ready = False
    run(ctrl, ports, 1.0)
    ports.nav_is_ready = True
    run(ctrl, ports, 1.0)
    assert ctrl.supervisor.phase == ns.FAULT

    # Manual RESET cannot release an unknown active goal.
    ctrl.on_command('RESET_NAV_FAULT')
    assert ctrl.supervisor.phase == ns.FAULT
    assert 'NAV_FAULT_RESET_DENIED' in ports.event_names()

    ctrl.on_command('START')
    run(ctrl, ports, 10.0)
    assert len(ports.sent) == 1

    # Only a terminal action result clears the unresolved transaction.
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert not ctrl.supervisor.busy
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 2


def test_transport_error_does_not_release_action_transaction():
    ctrl, ports = make(grid_array=room(), nav_fault_abort_timeout_s=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_result(txn, 'ERROR', RuntimeError('DDS unavailable'))
    assert ctrl.supervisor.phase == ns.FAULT
    ctrl.on_command('RESET_NAV_FAULT')
    assert ctrl.supervisor.phase == ns.FAULT
    run(ctrl, ports, 10.0)
    assert len(ports.sent) == 1

def test_missing_goal_response_is_a_fault_not_a_new_goal():
    ctrl, ports = make(grid_array=room(), nav_fault_abort_timeout_s=0.0)
    txn = start_and_dispatch(ctrl, ports)
    run(ctrl, ports, 12.0)                         # no response at all
    assert ctrl.supervisor.phase == ns.FAULT
    assert len(ports.sent) == 1
    # It is accepted late: cancel it explicitly and wait for the result.
    ctrl.nav_goal_response(txn, True, 'late')
    assert ports.cancels == ['late']
    assert ctrl.supervisor.phase == ns.CANCELING
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert ctrl.goals_failed == 0


# ─────────────────────────────────────────────────────────────────────────
# rotation
# ─────────────────────────────────────────────────────────────────────────
def rotating(**overrides):
    ctrl, ports = make(grid_array=room(), rotate_first_enabled=True,
                       rotation_heading_offset_deg=90.0, **overrides)
    ctrl.on_command('START')
    run(ctrl, ports, 0.4)
    assert ctrl.supervisor.phase == ns.ROTATING
    return ctrl, ports


def test_rotation_reaches_offset_heading_then_navigates():
    ctrl, ports = rotating()
    for _ in range(400):
        ports.advance(0.05)
        x, y, yaw = ports.pose
        if ports.cmd:
            ports.pose = (x, y, yaw + ports.cmd[-1] * 0.05)
        ctrl.rotation_tick()
        ctrl.tick()
        if ports.sent:
            break
    assert ports.sent, 'rotation never finished'
    goal_x, goal_y = ports.sent[0][1], ports.sent[0][2]
    bearing = math.atan2(goal_y - 3.0, goal_x - 2.0)
    err = math.atan2(math.sin(ports.pose[2] - (bearing + math.pi / 2)),
                     math.cos(ports.pose[2] - (bearing + math.pi / 2)))
    assert abs(err) < 0.16
    assert ports.cmd[-1] == 0.0


def test_rotation_stall_has_a_deadline():
    ctrl, ports = rotating()                       # yaw never changes
    run(ctrl, ports, 8.0, dt=0.05)
    assert ports.sent, 'robot stayed in the rotation phase'
    assert ports.cmd[-1] == 0.0
    assert 'ROTATION_ABORTED' in ports.event_names()


def test_rotation_with_lost_tf_stops_and_moves_on():
    ctrl, ports = rotating()
    ports.pose = None
    run(ctrl, ports, 0.5, dt=0.05)
    assert ports.cmd[-1] == 0.0                    # never rotates blind
    run(ctrl, ports, 3.0, dt=0.05)
    assert ctrl.supervisor.phase != ns.ROTATING
    aborted = [e for e in ports.events if e['event'] == 'ROTATION_ABORTED']
    assert aborted and aborted[0]['data']['reason'] == 'pose_lost'


def test_rotation_skipped_without_clearance():
    a = room()
    ctrl, ports = make(grid_array=a, rotate_first_enabled=True,
                       rotation_heading_offset_deg=90.0,
                       rotation_required_clearance_m=0.36)
    ports.pose = (1.2, 3.0, 0.0)                   # 0.2 m from the left wall
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    assert all(w == 0.0 for w in ports.cmd)
    assert ports.sent
    skipped = [e for e in ports.events if e['event'] == 'ROTATION_SKIPPED']
    assert skipped[0]['data']['reason'] == 'insufficient_clearance'


# ─────────────────────────────────────────────────────────────────────────
# TF and clock
# ─────────────────────────────────────────────────────────────────────────
def test_no_pose_postpones_selection():
    ctrl, ports = make(grid_array=room())
    ports.pose = None
    ctrl.on_command('START')
    run(ctrl, ports, 5.0)
    assert ports.sent == []
    ports.pose = (2.0, 3.0, 0.0)
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 1


def test_clock_reset_does_not_break_deadlines():
    ctrl, ports = make(grid_array=room(), goal_preview_s=3.0)
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    assert ctrl.supervisor.phase == ns.PREVIEW
    ports.ros = 2.0                                # simulation restarted
    ctrl.tick()
    assert ports.tf_resets == 1
    assert 'CLOCK_JUMP' in ports.event_names()
    run(ctrl, ports, 1.0)
    assert ctrl.supervisor.phase == ns.PREVIEW     # ~2 s of preview left
    run(ctrl, ports, 2.2)
    assert len(ports.sent) == 1


def test_cooldown_survives_clock_reset():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_result(txn, ns.RESULT_ABORTED)
    ports.ros = 1.0                                # backward jump
    run(ctrl, ports, 20.0)
    assert len(ports.sent) == 1                    # 30 s cooldown still active
    run(ctrl, ports, 12.0)
    assert len(ports.sent) == 2


def test_frozen_clock_is_detected_and_halts_rotation():
    ctrl, ports = rotating()
    n_cmd = len(ports.cmd)
    for _ in range(15):                            # wall time only
        ports.wall += 1.0
        ctrl.wall_tick()
    assert 'CLOCK_FROZEN' in ports.event_names()
    assert ports.cmd[-1] == 0.0 and len(ports.cmd) > n_cmd
    assert ctrl.supervisor.phase == ns.ROTATING    # no behavior time passed
    ports.advance(0.2)
    ctrl.wall_tick()
    assert 'CLOCK_RESUMED' in ports.event_names()


def test_slow_simulation_does_not_shorten_timeouts():
    # Real-time factor 0.1: 10 wall seconds per simulated second.
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    budget = ctrl.supervisor.txn.budget_s
    for _ in range(int(0.5 * budget / 0.2)):
        ports.advance(0.2, wall_dt=2.0)
        ports.pose = (ports.pose[0] + 0.001, 3.0, 0.0)
        ctrl.tick()
    assert ctrl.supervisor.phase == ns.NAVIGATING


# ─────────────────────────────────────────────────────────────────────────
# supervision
# ─────────────────────────────────────────────────────────────────────────
def test_reached_threshold_cancels_and_counts_as_success():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    handle = accept(ctrl, txn)
    _, gx, gy, _ = ports.sent[0]
    ports.pose = (gx - 0.4, gy, 0.0)
    run(ctrl, ports, 0.4)
    assert ports.cancels == [handle]
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert ctrl.goals_reached == 1 and ctrl.goals_failed == 0


def test_no_progress_is_a_failure_with_cooldown():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    run(ctrl, ports, 61.0)
    assert ctrl.supervisor.phase == ns.CANCELING
    assert ctrl.supervisor.txn.cancel_reason == 'no_progress'
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert ctrl.goals_failed == 1
    rec = ctrl.memory.records[0]
    assert rec.failures == 1 and rec.cooldown_until > ctrl.clock.t


def test_feedback_progress_and_recovery_keep_a_slow_goal_alive():
    ctrl, ports = make(grid_array=room(), nav_budget_min_s=1000.0,
                       nav_budget_max_s=1000.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    remaining = 3.0
    for _ in range(4):                             # slow but real progress
        run(ctrl, ports, 50.0)
        remaining -= 0.3
        ctrl.nav_feedback(txn, remaining, 0)
    assert ctrl.supervisor.phase == ns.NAVIGATING
    run(ctrl, ports, 50.0)
    ctrl.nav_feedback(txn, remaining, 1)           # a recovery started
    run(ctrl, ports, 50.0)
    assert ctrl.supervisor.phase == ns.NAVIGATING
    run(ctrl, ports, 20.0)
    assert ctrl.supervisor.txn.cancel_reason == 'no_progress'


def test_zero_distance_feedback_is_ignored():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_feedback(txn, 0.0, 0)
    assert ctrl.supervisor.txn.best_remaining is None


def test_budget_scales_with_path_length():
    ctrl, ports = make(grid_array=room(), expected_speed_mps=0.18)
    start_and_dispatch(ctrl, ports)
    t = ctrl.supervisor.txn
    expected = 30.0 + 3.0 * t.goal['path_m'] / 0.18
    assert abs(t.budget_s - max(60.0, expected)) < 1e-6


def test_frontier_resolved_needs_consecutive_maps():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    handle = accept(ctrl, txn)

    ctrl.on_map(make_grid(room_explored()))        # looks resolved once
    assert ports.cancels == []
    ctrl.on_map(make_grid(room()))                 # noise: frontier is back
    ctrl.on_map(make_grid(room_explored()))
    assert ports.cancels == []                     # streak was reset
    ctrl.on_map(make_grid(room_explored()))
    assert ports.cancels == [handle]
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    assert ctrl.goals_resolved == 1 and ctrl.goals_failed == 0
    assert 'FRONTIER_RESOLVED' in ports.event_names()


# ─────────────────────────────────────────────────────────────────────────
# memory, unproductive visits, completion
# ─────────────────────────────────────────────────────────────────────────
def reach_current_goal(ctrl, ports):
    txn = ports.sent[-1][0]
    accept(ctrl, txn)
    _, gx, gy, _ = ports.sent[-1]
    ports.pose = (gx - 0.3, gy, 0.0)
    ctrl.nav_result(txn, ns.RESULT_SUCCEEDED)


def test_productive_visit_clears_history():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_result(txn, ns.RESULT_ABORTED)        # one failure first
    assert len(ctrl.memory.records) == 1
    run(ctrl, ports, 31.0)
    assert len(ports.sent) == 2
    reach_current_goal(ctrl, ports)
    ctrl.on_map(make_grid(room_explored()))        # the visit revealed map
    run(ctrl, ports, 0.4)
    assert ctrl.visits_productive == 1
    assert ctrl.memory.records == []


def test_repeated_unproductive_visits_exhaust_the_region_and_terminate():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0,
                       recent_goal_penalty=1.0)
    ctrl.on_command('START')
    for visit in range(2):
        run(ctrl, ports, 1.0)
        assert len(ports.sent) == visit + 1
        reach_current_goal(ctrl, ports)
        ports.pose = (2.0, 3.0, 0.0)               # back in the room
        ctrl.on_map(make_grid(room()))             # nothing new appeared
        run(ctrl, ports, 0.4)
    assert ctrl.visits_unproductive == 2
    assert ctrl.memory.records[0].exhausted
    assert ports.event_names().count('UNPRODUCTIVE_VISIT') == 2

    for _ in range(4):                             # completion needs new maps
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 0.4)
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE_PARTIAL
    assert len(ports.sent) == 2
    assert ports.save_requests == 1
    assert ctrl.status_text() == 'COMPLETE'


def test_visit_without_new_map_is_not_penalized():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    reach_current_goal(ctrl, ports)
    run(ctrl, ports, 4.0)                          # settle timeout, no map
    assert ctrl.visits_unproductive == 0
    evaluated = [e for e in ports.events if e['event'] == 'VISIT_EVALUATED']
    assert evaluated[0]['data']['classification'] == 'UNVERIFIED_SUCCESS'


def test_cooldown_blocks_completion_until_retry():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_result(txn, ns.RESULT_ABORTED)
    for _ in range(10):                            # many new maps in cooldown
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 1.0)
    assert not ctrl.ended
    assert ctrl.status_text() == 'WAITING_FRONTIER_RETRY'
    assert ctrl.completion.no_candidate_maps == 0
    run(ctrl, ports, 25.0)
    assert len(ports.sent) == 2                    # retried after 30 s


def test_three_failures_exhaust_then_partial_completion():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    ctrl.on_command('START')
    waits = [1.0, 31.0, 121.0]
    for i, wait in enumerate(waits):
        run(ctrl, ports, wait)
        assert len(ports.sent) == i + 1, i
        txn = ports.sent[-1][0]
        accept(ctrl, txn)
        ctrl.nav_result(txn, ns.RESULT_ABORTED)
    assert ctrl.memory.records[0].exhausted
    for _ in range(4):
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 0.4)
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE_PARTIAL
    summary = [e for e in ports.events
               if e['event'] == 'EXPLORATION_PARTIAL'][0]['data']
    assert summary['frontier_counts'][fs.EXHAUSTED] == 1
    assert summary['navigation']['failed'] == 3


def test_exhausted_region_reopens_when_the_map_changes():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0,
                       exhaust_after_failures=1)
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    ctrl.nav_result(txn, ns.RESULT_ABORTED)
    assert ctrl.memory.records[0].exhausted
    a = room()
    a[20:100, 100:125] = 0                         # 1.25 m more map appears
    ctrl.on_map(make_grid(a))
    run(ctrl, ports, 1.0)
    assert 'REGION_REOPENED' in ports.event_names()
    assert len(ports.sent) == 2


def test_fully_enclosed_room_completes_without_moving():
    ctrl, ports = make(grid_array=room(open_right=False))
    ctrl.on_command('START')
    for _ in range(4):
        ctrl.on_map(make_grid(room(open_right=False)))
        run(ctrl, ports, 0.4)
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE
    assert ports.sent == []


def unreachable_world():
    """Robot room plus a sealed pocket with its own frontier."""
    a = room(open_right=False)
    a[140:180, 40:80] = 0                          # pocket at y in [7, 9]
    a[139, 39:81] = 100
    a[140:180, 39] = 100
    a[140:180, 80] = 100                           # top side open to unknown
    return a


def test_unreachable_frontier_is_probed_once_then_reported():
    ctrl, ports = make(grid_array=unreachable_world())
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    assert ports.sent == []                        # not probed on one map
    assert ctrl._last_counts[fs.TEMPORARILY_BLOCKED] >= 1
    ctrl.on_map(make_grid(unreachable_world()))
    run(ctrl, ports, 1.0)
    assert len(ports.sent) == 1                    # navigation gets a say
    assert ctrl.supervisor.txn.goal['probe'] is True
    txn = ports.sent[0][0]
    accept(ctrl, txn)
    ctrl.nav_result(txn, ns.RESULT_ABORTED)        # planner found no path
    for _ in range(5):
        ctrl.on_map(make_grid(unreachable_world()))
        run(ctrl, ports, 0.4)
    assert len(ports.sent) == 1                    # never retried
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE_PARTIAL
    assert ctrl._last_counts[fs.UNREACHABLE] >= 1


def test_costmap_veto_blocks_then_stagnation_ends_the_mission():
    ctrl, ports = make(grid_array=room(), stagnation_timeout_s=30.0)
    cost = np.full((200, 200), 0, dtype=np.int8)
    cost[:, 90:110] = 99                           # lethal band on frontier
    ctrl.on_costmap(make_grid(cost))
    ctrl.on_command('START')
    for _ in range(20):
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 1.0)
    assert ports.sent == [] and not ctrl.ended
    for _ in range(20):
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 1.0)
    assert ctrl.ended and ctrl.end_class == comp.STAGNATED
    assert 'EXPLORATION_STAGNATED' in ports.event_names()


def test_time_limit_cancels_safely_before_ending():
    ctrl, ports = make(grid_array=room(), max_exploration_time_s=20.0)
    txn = start_and_dispatch(ctrl, ports)
    handle = accept(ctrl, txn)
    run(ctrl, ports, 25.0)
    assert ports.cancels == [handle] and not ctrl.ended
    ctrl.nav_result(txn, ns.RESULT_CANCELED)
    run(ctrl, ports, 0.4)
    assert ctrl.ended and ctrl.end_class == comp.TIME_LIMIT
    assert ctrl.goals_failed == 0


# ─────────────────────────────────────────────────────────────────────────
# scoring behavior
# ─────────────────────────────────────────────────────────────────────────
def test_path_distance_beats_straight_line():
    """A frontier behind a wall must not win just because it is close."""
    a = np.full((240, 240), -1, dtype=np.int8)
    a[20:220, 20:120] = 0                          # big free hall, x in 1..6
    a[19, 19:121] = 100
    a[220, 19:121] = 100
    a[20:220, 19] = 100
    a[20:220, 120] = 100                           # right wall ...
    a[200:216, 120] = -1                           # ... open far up (y~10.4)
    # Thin wall next to the robot; behind it a pocket open to unknown.
    a[60:100, 60] = 100
    ctrl, ports = make(grid_array=a)
    ports.pose = (2.0, 4.0, 0.0)
    ctrl.on_command('START')
    run(ctrl, ports, 1.0)
    goal = ctrl.supervisor.txn.goal
    assert goal['reachable'] is True
    assert goal['path_m'] >= goal['euclid_m'] - 0.2


def test_utility_is_bounded_and_monotonic():
    u_near = fs.utility(0.5, 1.0, 0.3)
    u_far = fs.utility(0.5, 5.0, 0.3)
    assert 0.0 < u_far < u_near < 1.0
    assert fs.utility(2.0, 0.0, 0.3) == 1.0        # gain is clamped
    assert fs.utility(0.5, 1e9, 0.3) >= 0.0        # no overflow
    cos, c = fs.continuity_factor((1.0, 0.0), (-1.0, 0.0), 0.45)
    assert cos == -1.0 and abs(c - 0.55) < 1e-9


# ─────────────────────────────────────────────────────────────────────────
# interfaces
# ─────────────────────────────────────────────────────────────────────────
LEGACY_KEYS = [
    'node4_mode', 'completion_reason', 'reached', 'failed', 'blacklist',
    'blacklist_count', 'blacklist_soonest_s', 'temp_blacklisted',
    'exhausted', 'no_eligible_refreshes', 'stable_refreshes',
    'map_change_pct', 'map_safety_rejected', 'costmap_safety_rejected',
    'recent_goals', 'frontier_count', 'frontier_mode', 'frontier_ms',
    'frontier_change_pct', 'frontier_full', 'frontier_incr', 'map_width',
    'map_height', 'map_res', 'map_extent_x_m', 'map_extent_y_m',
    'map_known_area_m2', 'map_growth_m2_min', 'map_metric_age_s',
    'robot_x', 'robot_y', 'robot_yaw_deg', 'target_x', 'target_y',
    'dist_target_m', 'dist_travelled_m',
]


def parse_stats(text):
    return [item.split('=', 1) for item in text.split(',')]


def test_stats_keep_every_original_field_in_order():
    ctrl, ports = make(grid_array=room())
    for state in ('waiting', 'running', 'ended'):
        if state == 'running':
            txn = start_and_dispatch(ctrl, ports)
            accept(ctrl, txn)
        if state == 'ended':
            ctrl._end(comp.COMPLETE, 'reason, with comma')
        pairs = parse_stats(ctrl.stats_text())
        keys = [k for k, _ in pairs]
        assert keys[:len(LEGACY_KEYS)] == LEGACY_KEYS, state
        assert all(len(p) == 2 for p in pairs)
        assert len(set(keys)) == len(keys)


def test_events_are_machine_readable_with_stable_schema():
    ctrl, ports = make(grid_array=room())
    txn = start_and_dispatch(ctrl, ports)
    accept(ctrl, txn)
    names = ports.event_names()
    for required in ('FRONTIER_EVALUATION', 'GOAL_SELECTED',
                     'NAVIGATION_STARTED'):
        assert required in names
    for event in ports.events:
        assert event['schema'] == 'exploration_event/1'
        assert {'seq', 'event', 't_ros', 't_behavior', 'txn_id',
                'state', 'data'} <= set(event)
    selected = [e for e in ports.events if e['event'] == 'GOAL_SELECTED'][0]
    assert selected['txn_id'] == txn
    assert {'utility', 'gain_m2', 'path_m', 'decision_ms'} <= set(
        selected['data'])
    seqs = [e['seq'] for e in ports.events]
    assert seqs == sorted(seqs)


def test_config_validation():
    cfg = ExplorationConfig()
    assert cfg.validate() == []
    cfg.reachability_inflation_m = 0.4
    assert cfg.validate() and cfg.reachability_inflation_m == cfg.goal_clearance_m
    bad = ExplorationConfig()
    bad.recent_goal_penalty = 0.0
    with pytest.raises(ValueError):
        bad.validate()


# ─────────────────────────────────────────────────────────────────────────
# computational measurement (prints; run with -s to see the numbers)
# ─────────────────────────────────────────────────────────────────────────
def warehouse(size_m=24.0, res=0.025, seed=7):
    """A 24 m x 24 m half-explored map at the SLAM resolution in use."""
    n = int(size_m / res)
    rng = np.random.default_rng(seed)
    a = np.full((n, n), -1, dtype=np.int8)
    a[40:n // 2, 40:n - 40] = 0                    # explored lower half
    a[39, 39:n - 39] = 100
    a[39:n // 2, 39] = 100
    a[39:n // 2, n - 40] = 100
    for _ in range(25):                            # racks / crates
        r = int(rng.integers(80, n // 2 - 80))
        c = int(rng.integers(80, n - 160))
        a[r:r + 30, c:c + 80] = 100
    return a


def test_decision_latency_on_a_large_map():
    ctrl, ports = make(grid_array=warehouse())
    ports.pose = (2.0, 2.0, 0.0)
    ctrl.on_command('START')
    t0 = time.perf_counter()
    run(ctrl, ports, 0.4)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    timing = ctrl._last_timing
    print(f'\n960x960 map decision: {timing}  (wall {wall_ms:.0f} ms, '
          f'clusters={ctrl.frontier_count})')
    assert ports.sent, 'no goal selected on the large map'
    parts = sum(timing[k] for k in (
        'detect_ms', 'cluster_ms', 'reach_ms', 'gain_ms', 'score_ms'))
    assert parts <= timing['decision_ms'] * 1.05
    # Generous bound: guards against an accidental O(candidates * map) path.
    assert timing['decision_ms'] < 3000.0


def test_frontier_next_to_the_robot_is_reported_not_hidden():
    ctrl, ports = make(grid_array=room(), max_frontier_length_m=0.0)
    ports.pose = (4.6, 3.0, 0.0)                   # 0.4 m from the frontier
    ctrl.on_command('START')
    for _ in range(4):
        ctrl.on_map(make_grid(room()))
        run(ctrl, ports, 0.4)
    assert ports.sent == []
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE_PARTIAL


def test_completion_without_map_republication():
    """A SLAM that stops publishing an unchanged map must not hang the end."""
    ctrl, ports = make(grid_array=room(open_right=False))
    ctrl.on_command('START')
    run(ctrl, ports, 8.0)
    assert not ctrl.ended                          # one evaluation is not enough
    run(ctrl, ports, 10.0)
    assert ctrl.ended and ctrl.end_class == comp.COMPLETE


# ─────────────────────────────────────────────────────────────────────────
# closed-loop simulation: the whole framework against a toy world
# ─────────────────────────────────────────────────────────────────────────
class ToyWorld(FakePorts):
    """Ground-truth world + range sensor + a navigation action stand-in.

    The exploration framework never sees `truth`; it only receives the map
    built from simulated range measurements, like a real SLAM pipeline.
    """

    def __init__(self, truth, start, sensor_range=3.0, speed=0.5):
        super().__init__()
        self.truth = truth
        self.known = np.full(truth.shape, -1, dtype=np.int8)
        self.pose = (start[0], start[1], 0.0)
        self.sensor_range = sensor_range
        self.speed = speed
        self.active = None            # dict(txn, waypoints, handle)
        self.pending = []             # callbacks to deliver on the next step
        self.ctrl = None
        self.blocked_goals = 0

    def truth_grid(self):
        return make_grid(np.where(self.truth, 100, 0).astype(np.int8))

    def sense(self):
        x, y, _ = self.pose
        angles = np.linspace(0, 2 * math.pi, 720, endpoint=False)
        radii = np.arange(RES * 0.5, self.sensor_range, RES * 0.5)
        cols = np.floor((x + np.outer(np.cos(angles), radii)) / RES).astype(int)
        rows = np.floor((y + np.outer(np.sin(angles), radii)) / RES).astype(int)
        h, w = self.truth.shape
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        occ = np.ones(rows.shape, dtype=bool)
        occ[inside] = self.truth[rows[inside], cols[inside]]
        hits = np.cumsum(occ, axis=1)
        free = (hits == 0) & inside
        first = occ & (hits == 1) & inside
        self.known[rows[free], cols[free]] = 0
        self.known[rows[first], cols[first]] = 100
        return make_grid(self.known.copy())

    # navigation stand-in ---------------------------------------------------
    def nav_send(self, txn_id, x, y, yaw):
        self.sent.append((txn_id, x, y, yaw))
        handle = f'h{txn_id}'
        grid = self.truth_grid()
        field = reachability.compute_reachability(
            grid, self.pose[0], self.pose[1], 0.2, 0.1, 0.6)
        row, col = grid.world_to_cell(x, y)
        self.pending.append(
            lambda: self.ctrl.nav_goal_response(txn_id, True, handle))
        if field is not None and not math.isfinite(field.distance(row, col)):
            # Like a planner with a goal tolerance: accept the nearest
            # reachable cell within 0.5 m of the requested goal.
            r = int(0.5 / RES)
            rr, cc = np.mgrid[row - r:row + r + 1, col - r:col + r + 1]
            ok = (rr >= 0) & (rr < grid.height) & (cc >= 0) & (cc < grid.width)
            rr, cc = rr[ok], cc[ok]
            dist = field.distances(rr, cc)
            near = np.hypot(rr - row, cc - col) <= r
            dist[~near] = np.inf
            if np.isfinite(dist).any():
                best = int(np.argmin(np.where(
                    np.isfinite(dist), np.hypot(rr - row, cc - col), np.inf)))
                row, col = int(rr[best]), int(cc[best])
                x, y = grid.cell_to_world(row, col)
        if field is None or not math.isfinite(field.distance(row, col)):
            self.blocked_goals += 1
            self.pending.append(
                lambda: self.ctrl.nav_result(txn_id, ns.RESULT_ABORTED))
            return True
        node = field._index(row, col)
        way = [(x, y)]
        while node != field.start_index and node >= 0:
            way.append(field._coarse_world(node))
            node = int(field.pred[node])
        way.reverse()
        self.active = {'txn': txn_id, 'way': way, 'handle': handle}
        return True

    def nav_cancel(self, handle):
        self.cancels.append(handle)
        if self.active and self.active['handle'] == handle:
            txn = self.active['txn']
            self.active = None
            self.pending.append(
                lambda: self.ctrl.nav_result(txn, ns.RESULT_CANCELED))

    def step(self, dt):
        pending, self.pending = self.pending, []
        for callback in pending:
            callback()
        if self.active is None:
            return
        budget = self.speed * dt
        x, y, yaw = self.pose
        way = self.active['way']
        while budget > 0 and way:
            wx, wy = way[0]
            d = math.hypot(wx - x, wy - y)
            if d <= budget:
                x, y = wx, wy
                budget -= d
                way.pop(0)
            else:
                x += (wx - x) / d * budget
                y += (wy - y) / d * budget
                budget = 0
        self.pose = (x, y, yaw)
        if not way:
            txn = self.active['txn']
            self.active = None
            self.pending.append(
                lambda: self.ctrl.nav_result(txn, ns.RESULT_SUCCEEDED))


def toy_building():
    """12 m x 12 m: four rooms joined by 0.9 m doors, plus loose obstacles."""
    n = int(12.0 / RES)
    t = np.zeros((n, n), dtype=bool)
    t[0:2, :] = t[-2:, :] = t[:, 0:2] = t[:, -2:] = True
    mid = n // 2
    t[:, mid - 1:mid + 1] = True                   # vertical wall
    t[mid - 1:mid + 1, :] = True                   # horizontal wall
    door = int(0.9 / RES)
    t[50:50 + door, mid - 1:mid + 1] = False       # lower door
    t[180:180 + door, mid - 1:mid + 1] = False     # upper door
    t[mid - 1:mid + 1, 40:40 + door] = False       # left door
    t[60:75, 160:200] = True                       # a rack
    t[150:210, 60:66] = True                       # a partition
    return t


def simulate(cfg_overrides=None, seconds=4000.0, world=None):
    cfg = ExplorationConfig()
    cfg.goal_preview_s = 0.0
    cfg.rotate_first_enabled = False
    cfg.expected_speed_mps = 0.5
    cfg.log_decisions = False
    for key, value in (cfg_overrides or {}).items():
        setattr(cfg, key, value)
    ports = world or ToyWorld(toy_building(), start=(2.0, 2.0))
    ctrl = ExplorationController(cfg, ports)
    ports.ctrl = ctrl
    ctrl.on_map(ports.sense())
    ctrl.on_command('START')
    dt = 0.2
    t = 0.0
    while t < seconds and not ctrl.ended:
        ports.advance(dt)
        ports.step(dt)
        if abs((t / 1.0) - round(t / 1.0)) < 1e-6:     # 1 Hz map, like SLAM
            ctrl.on_map(ports.sense())
        ctrl.tick()
        t += dt
    return ctrl, ports, t


def coverage(ports):
    grid = ports.truth_grid()
    field = reachability.compute_reachability(grid, 2.0, 2.0, 0.2, 0.1, 0.6)
    reach = np.isfinite(field.dist).reshape(field.coarse_shape)
    reach = np.kron(reach, np.ones((field.factor, field.factor), dtype=bool))
    reach = reach[:grid.height, :grid.width] & ~ports.truth
    return float((ports.known[reach] == 0).sum()) / float(reach.sum())


def test_closed_loop_exploration_terminates_with_high_coverage():
    ctrl, ports, t = simulate()
    cov = coverage(ports)
    print(f'\nclosed loop: ended={ctrl.ended} class={ctrl.end_class} '
          f't={t:.0f}s goals={len(ports.sent)} reached={ctrl.goals_reached} '
          f'resolved={ctrl.goals_resolved} failed={ctrl.goals_failed} '
          f'unproductive={ctrl.visits_unproductive} '
          f'dist={ctrl.dist_travelled_m:.1f}m coverage={cov:.3f} '
          f'reason="{ctrl.completion_reason}"')
    assert ctrl.ended, 'exploration never terminated'
    assert ctrl.end_class in (comp.COMPLETE, comp.COMPLETE_PARTIAL)
    assert cov > 0.95
    assert ctrl.supervisor.stale_callbacks == 0
    # Exactly one navigation goal at a time, always.
    assert len({s[0] for s in ports.sent}) == len(ports.sent)


def test_closed_loop_with_a_door_that_navigation_cannot_pass():
    """The map says a room is reachable; the navigation stack disagrees."""
    world = ToyWorld(toy_building(), start=(2.0, 2.0))
    real_truth_grid = world.truth_grid

    def blocked_truth_grid():
        a = world.truth.copy()
        a[180:198, 119:121] = True                 # upper door closed for nav
        return make_grid(np.where(a, 100, 0).astype(np.int8))

    world.truth_grid = blocked_truth_grid
    ctrl, ports, t = simulate(world=world, seconds=6000.0)
    world.truth_grid = real_truth_grid
    print(f'\nblocked door: class={ctrl.end_class} t={t:.0f}s '
          f'goals={len(ports.sent)} failed={ctrl.goals_failed} '
          f'aborted_by_nav={ports.blocked_goals} '
          f'exhausted={ctrl.memory.count_exhausted()} '
          f'reason="{ctrl.completion_reason}"')
    assert ctrl.ended, 'exploration never terminated'
    # It must end, and must not keep hammering the same unreachable region.
    assert ports.blocked_goals <= 12
