#!/usr/bin/env python3
"""
exploration_manager.py — Frontera (ROS 2 node)
=============================================================
Supervised frontier-exploration framework for any ROS 2 mobile robot that has

  * a SLAM system publishing nav_msgs/OccupancyGrid,
  * the robot pose available through TF (map -> base), and
  * a nav2_msgs/NavigateToPose action server (Nav2 or compatible).

This file is only the ROS adapter: parameters, topics, TF and the action
client. All decision logic lives in the `exploration` package next to it and
has no ROS dependency (see exploration/__init__.py for the module map).

Configuration: every setting is a ROS 2 parameter documented in
config/frontera.yaml. Nothing robot-specific is
hard-coded here.

State machine (unchanged external behavior):
  WAITING -> START -> RUNNING <-> PAUSE -> PAUSED
                              -> STOP  -> STOPPED

Key design:
- Frontier selection happens ONLY in the controller's tick() gate.
  Callbacks never select; they only set flags.
- The node is safe to launch during system startup: nothing can move the
  robot until START is received on the control topic.
- Each goal is a transaction with an ID. Late navigation callbacks are
  matched to their transaction and cannot affect a newer goal.
- A goal is over only when the navigation stack confirms it. An unconfirmed
  cancellation blocks new goals and is reported, never assumed.
- Behavior durations use ROS (simulation) time; wall time is used only to
  detect a frozen clock and unresponsive components.
- STOP does not shut down other processes. The program manager owns process
  lifecycle; this node only stops exploring and reports STOPPED.
- Map files are written by the program manager on request, never here.
"""
# Author: Raymundo Gomez

import importlib
import math
import os
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from tf2_ros import Buffer, TransformListener

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray

# The framework package. To reuse this node in another ROS 2 package, copy
# the `exploration` folder there and point this at it (or set the
# EXPLORATION_FRAMEWORK_PACKAGE environment variable).
FRAMEWORK_PACKAGE = os.environ.get(
    'EXPLORATION_FRAMEWORK_PACKAGE', 'frontera')

_config = importlib.import_module(FRAMEWORK_PACKAGE + '.config')
_controller = importlib.import_module(FRAMEWORK_PACKAGE + '.controller')
_grid = importlib.import_module(FRAMEWORK_PACKAGE + '.grid')
_scoring = importlib.import_module(FRAMEWORK_PACKAGE + '.frontier_scoring')
_supervisor = importlib.import_module(
    FRAMEWORK_PACKAGE + '.navigation_supervisor')

ExplorationConfig = _config.ExplorationConfig
ExplorationController = _controller.ExplorationController
GridMap = _grid.GridMap

_RESULT_NAMES = {
    GoalStatus.STATUS_SUCCEEDED: _supervisor.RESULT_SUCCEEDED,
    GoalStatus.STATUS_CANCELED: _supervisor.RESULT_CANCELED,
    GoalStatus.STATUS_ABORTED: _supervisor.RESULT_ABORTED,
}

# Marker colors per candidate class (r, g, b).
_CLASS_COLORS = {
    _scoring.ELIGIBLE: (0.1, 0.8, 0.2),
    _scoring.COOLDOWN: (1.0, 0.9, 0.1),
    _scoring.TEMPORARILY_BLOCKED: (0.2, 0.6, 1.0),
    _scoring.UNREACHABLE: (0.6, 0.2, 0.8),
    _scoring.EXHAUSTED: (0.9, 0.1, 0.1),
    _scoring.LOW_INFORMATION_GAIN: (0.6, 0.6, 0.6),
    _scoring.UNSAFE: (0.3, 0.3, 0.3),
}


def _coerce(name, value, default):
    """Convert a parameter value to the type of its default."""
    try:
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, str):
            return '' if value is None else str(value)
        if isinstance(default, list):
            if value is None:
                return list(default)
            if isinstance(value, (str, bytes)):
                value = [value]
            item_type = type(default[0]) if default else str
            return [item_type(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f'parameter "{name}": cannot use {value!r}') from exc
    return value


class ExplorationManager(Node):

    def __init__(self):
        super().__init__('exploration_manager')

        # ── parameters -> configuration ───────────────────────────────────
        self.cfg = self._load_config()
        cfg = self.cfg

        self._global_frame = cfg.global_frame.strip().lstrip('/')
        self._odom_child_frame = ''
        self._odom_frame_from_msg = ''
        self._last_pose_reason = ''
        self._orientation_warned = False
        self._wall_log = {}

        # ── TF ────────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        # Latest-transform lookups never block, so the listener can share
        # this node's executor.
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=False)

        # ── QoS ───────────────────────────────────────────────────────────
        latched = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE)
        map_qos = latched if cfg.map_qos_transient_local else QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.RELIABLE)

        # ── publishers ────────────────────────────────────────────────────
        # The current target is state, not a stream: latched so a viewer
        # that connects late still receives it.
        self.goal_pub = self.create_publisher(
            PoseStamped, cfg.goal_topic, latched)
        self.frontier_pub = self.create_publisher(
            MarkerArray, cfg.frontiers_topic, latched)
        self.status_pub = self.create_publisher(String, cfg.status_topic, 10)
        self.stats_pub = self.create_publisher(String, cfg.stats_topic, 10)
        self.events_pub = self.create_publisher(String, cfg.events_topic, 100)
        self.cmd_vel_pub = self.create_publisher(Twist, cfg.cmd_vel_topic, 10)
        self.map_save_status_pub = self.create_publisher(
            String, cfg.map_save_status_topic, 10)
        self.map_save_request_pub = self.create_publisher(
            String, cfg.map_save_request_topic, 10)
        self.sensor_pub = None
        if cfg.sensor_look_topic:
            self.sensor_pub = self.create_publisher(
                Float32, cfg.sensor_look_topic, 10)

        # ── navigation action client ──────────────────────────────────────
        self.nav_client = ActionClient(
            self, NavigateToPose, cfg.navigate_action)

        # ── controller (all decision logic) ───────────────────────────────
        self.controller = ExplorationController(cfg, self)

        # ── subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            OccupancyGrid, cfg.map_topic,
            self._guarded('map_cb', self._map_cb), map_qos)
        if cfg.costmap_topic:
            self.create_subscription(
                OccupancyGrid, cfg.costmap_topic,
                self._guarded('costmap_cb', self._costmap_cb), 10)
        if cfg.odom_topic:
            self.create_subscription(
                Odometry, cfg.odom_topic,
                self._guarded('odom_cb', self._odom_cb), 10)
        self.create_subscription(
            String, cfg.control_topic,
            self._guarded('control_cb', self._control_cb), 10)

        # ── timers ────────────────────────────────────────────────────────
        # ROS-time timers: they follow simulation time.
        self.create_timer(
            1.0 / cfg.decision_rate_hz,
            self._guarded('tick', self.controller.tick))
        self.create_timer(
            1.0 / cfg.rotation_control_rate_hz,
            self._guarded('rotation_tick', self.controller.rotation_tick))
        self.create_timer(
            1.0 / cfg.status_rate_hz,
            self._guarded('status_tick', self.controller.status_tick))
        # Wall-clock timer: keeps firing when simulation time is frozen.
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(
            1.0, self._guarded('wall_tick', self.controller.wall_tick),
            clock=self._steady_clock)

        self.get_logger().info(
            f'Frontera ready (WAITING). map={cfg.map_topic} '
            f'base={cfg.robot_base_frame} action={cfg.navigate_action} '
            f'cmd_vel={cfg.cmd_vel_topic}')
        self.get_logger().info(
            f'Send START on {cfg.control_topic} to begin. The robot cannot '
            'move before that.')

    # =====================================================================
    # configuration
    # =====================================================================
    def _load_config(self):
        cfg = ExplorationConfig()
        descriptor = ParameterDescriptor(dynamic_typing=True)
        for name, default in ExplorationConfig.field_defaults().items():
            self.declare_parameter(name, default, descriptor)
            value = self.get_parameter(name).value
            setattr(cfg, name, _coerce(name, value, default))

        cfg.robot_base_frame = cfg.robot_base_frame.strip().lstrip('/')
        if not cfg.map_topic:
            cfg.map_topic = '/map'
        return cfg

    def _guarded(self, name, fn):
        """Wrap a callback so one exception cannot kill the node.

        An exception escaping any ROS callback ends rclpy.spin() and with it
        the node. If that happened while a goal is active, nothing would be
        left to supervise the robot. The error is logged and the next
        callback runs normally. Used for every timer, subscription and
        action callback.
        """
        def wrapper(*args):
            try:
                return fn(*args)
            except Exception as exc:
                try:
                    self._throttled(
                        'exc_' + name, 2.0, 'error',
                        f'{name} raised {type(exc).__name__}: {exc}')
                except Exception:
                    pass
            return None
        return wrapper

    # =====================================================================
    # ROS callbacks -> controller
    # =====================================================================
    def _grid_from_msg(self, msg):
        info = msg.info
        q = info.origin.orientation
        if (abs(q.x) > 1e-3 or abs(q.y) > 1e-3 or abs(q.z) > 1e-3) \
                and not self._orientation_warned:
            self._orientation_warned = True
            self.get_logger().warning(
                'Occupancy grid origin is rotated; this framework assumes an '
                'axis-aligned grid and positions will be wrong')
        return GridMap(
            msg.data, info.width, info.height, info.resolution,
            info.origin.position.x, info.origin.position.y,
            frame_id=msg.header.frame_id.strip().lstrip('/') or 'map',
            free_max=self.cfg.map_free_max_value,
            occupied_min=self.cfg.map_occupied_min_value)

    def _map_cb(self, msg):
        try:
            grid = self._grid_from_msg(msg)
        except Exception as exc:
            self._throttled('bad_map', 5.0, 'warn', f'Ignoring map: {exc}')
            return
        if not self.cfg.global_frame:
            self._global_frame = grid.frame_id
        self.controller.on_map(grid)

    def _costmap_cb(self, msg):
        try:
            self.controller.on_costmap(self._grid_from_msg(msg))
        except Exception as exc:
            self._throttled(
                'bad_costmap', 5.0, 'warn', f'Ignoring costmap: {exc}')

    def _odom_cb(self, msg):
        self._odom_child_frame = str(msg.child_frame_id).strip().lstrip('/')
        self._odom_frame_from_msg = str(msg.header.frame_id).strip().lstrip('/')

    def _control_cb(self, msg):
        self.controller.on_command(msg.data)

    def _throttled(self, key, period_s, level, text):
        now = time.monotonic()
        if now - self._wall_log.get(key, -1e18) >= period_s:
            self._wall_log[key] = now
            self.log(level, text)

    # =====================================================================
    # ports used by the controller
    # =====================================================================
    def ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def wall_now(self):
        return time.monotonic()

    def log(self, level, text):
        # rclpy caches the severity of each logging call site (file + line)
        # and raises if one line is used with different severities. Each
        # severity therefore needs its own call line here.
        logger = self.get_logger()
        if level == 'error':
            logger.error(text)
        elif level in ('warn', 'warning'):
            logger.warning(text)
        elif level == 'debug':
            logger.debug(text)
        else:
            logger.info(text)

    # ── pose ─────────────────────────────────────────────────────────────
    def _base_frames(self):
        frames = [self.cfg.robot_base_frame]
        if self.cfg.use_odom_child_frame and self._odom_child_frame:
            frames.append(self._odom_child_frame)
        frames.extend(self.cfg.robot_base_frame_fallbacks)
        unique = []
        for frame in frames:
            frame = str(frame).strip().lstrip('/')
            if frame and frame not in unique:
                unique.append(frame)
        return unique

    def _stamp_problem(self, tr, now_s, max_age_s):
        """Return None if the transform's stamp is acceptable, else why."""
        stamp = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            return None if self.cfg.tf_accept_zero_stamp else 'unstamped'
        age = now_s - stamp
        if max_age_s > 0.0 and age > max_age_s:
            return f'stale ({age:.2f} s old)'
        if age < -self.cfg.tf_future_tolerance_s:
            return f'stamped {-age:.2f} s in the future'
        return None

    @staticmethod
    def _planar(tr):
        """(x, y, yaw) of a transform, or None if it is not finite."""
        t = tr.transform.translation
        q = tr.transform.rotation
        if not all(math.isfinite(v) for v in (t.x, t.y, q.x, q.y, q.z, q.w)):
            return None
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(t.x), float(t.y), float(yaw)

    def _odom_frame_name(self):
        frame = self.cfg.odom_frame.strip().lstrip('/')
        return frame or self._odom_frame_from_msg

    def get_pose(self):
        """Fresh (x, y, yaw) of the robot in the map frame, or None.

        1. Direct lookup map -> base, accepted if at most tf_max_age_s old.
        2. Otherwise, if the odometry frame is known, map -> odom and
           odom -> base are looked up separately and combined. Only
           odom -> base must be fresh (tf_max_age_s): the SLAM correction
           map -> odom changes slowly and many SLAM systems only republish
           it when they process a new scan, so while the robot stands still
           its stamp stops advancing. Its age is limited by
           tf_map_odom_max_age_s (0 = no limit).
        Missing, stale, extrapolated or badly stamped transforms return None,
        and callers must cope with that.
        """
        target = self._global_frame or 'map'
        now_s = self.ros_now()
        reason = 'missing'
        cfg = self.cfg

        for frame in self._base_frames():
            try:
                tr = self.tf_buffer.lookup_transform(target, frame, Time())
            except Exception as exc:
                reason = f'lookup failed: {type(exc).__name__}'
            else:
                problem = self._stamp_problem(tr, now_s, cfg.tf_max_age_s)
                pose = self._planar(tr) if problem is None else None
                if pose is not None:
                    self._last_pose_reason = ''
                    return pose
                reason = problem or 'non-finite'

            odom = self._odom_frame_name()
            if not odom or odom in (target, frame):
                continue
            try:
                t_mo = self.tf_buffer.lookup_transform(target, odom, Time())
                t_ob = self.tf_buffer.lookup_transform(odom, frame, Time())
            except Exception as exc:
                reason += f'; split lookup failed: {type(exc).__name__}'
                continue
            problem = (
                self._stamp_problem(t_ob, now_s, cfg.tf_max_age_s) or
                self._stamp_problem(t_mo, now_s, cfg.tf_map_odom_max_age_s))
            if problem is not None:
                reason += f'; split: {problem}'
                continue
            mo = self._planar(t_mo)
            ob = self._planar(t_ob)
            if mo is None or ob is None:
                reason += '; split: non-finite'
                continue
            # Planar composition map->base = (map->odom) * (odom->base).
            c, s_ = math.cos(mo[2]), math.sin(mo[2])
            x = mo[0] + c * ob[0] - s_ * ob[1]
            y = mo[1] + s_ * ob[0] + c * ob[1]
            yaw = math.atan2(math.sin(mo[2] + ob[2]), math.cos(mo[2] + ob[2]))
            self._last_pose_reason = ''
            return x, y, yaw

        self._last_pose_reason = reason
        if self.controller.grid is not None:    # quiet during system startup
            self._throttled(
                'no_pose', 5.0, 'warn',
                f'No usable TF {target} -> robot base ({reason}); tried: '
                f'{", ".join(self._base_frames())}')
        return None

    def reset_tf(self):
        try:
            self.tf_buffer.clear()
        except Exception as exc:
            self.log('warn', f'TF buffer clear failed: {exc}')

    # ── navigation action ────────────────────────────────────────────────
    def nav_ready(self):
        return self.nav_client.server_is_ready()

    def _goal_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self._global_frame or 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = 0.0
        # Arrival heading = heading at dispatch (keeps the side chosen by
        # the rotate-first phase on an omnidirectional base).
        pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
        pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
        return pose

    def nav_send(self, txn_id, x, y, yaw):
        goal = NavigateToPose.Goal()
        goal.pose = self._goal_pose(x, y, yaw)
        try:
            on_feedback = self._guarded('nav_feedback', self._nav_feedback)
            on_response = self._guarded('nav_response', self._nav_response)
            future = self.nav_client.send_goal_async(
                goal,
                feedback_callback=lambda fb, t=txn_id: on_feedback(t, fb))
            future.add_done_callback(
                lambda f, t=txn_id: on_response(t, f))
        except Exception as exc:
            self.log('error', f'send_goal failed: {exc}')
            return False
        return True

    def _nav_response(self, txn_id, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.controller.nav_goal_response(txn_id, False, None, exc)
            return
        if handle is None or not handle.accepted:
            self.controller.nav_goal_response(txn_id, False, None, None)
            return
        # Register the result callback before telling the controller, which
        # may cancel the goal immediately.
        on_result = self._guarded('nav_result', self._nav_result)
        handle.get_result_async().add_done_callback(
            lambda f, t=txn_id: on_result(t, f))
        self.controller.nav_goal_response(txn_id, True, handle, None)

    def _nav_feedback(self, txn_id, feedback_msg):
        fb = feedback_msg.feedback
        self.controller.nav_feedback(
            txn_id,
            getattr(fb, 'distance_remaining', float('nan')),
            getattr(fb, 'number_of_recoveries', 0))

    def _nav_result(self, txn_id, future):
        try:
            status = future.result().status
        except Exception as exc:
            self.controller.nav_result(txn_id, 'ERROR', exc)
            return
        self.controller.nav_result(
            txn_id, _RESULT_NAMES.get(status, f'STATUS_{status}'), None)

    def nav_cancel(self, handle):
        if handle is None:
            return
        try:
            handle.cancel_goal_async()
        except Exception as exc:
            self.log('error', f'cancel request failed: {exc}')

    # ── outputs ──────────────────────────────────────────────────────────
    def cmd_rotate(self, angular_z):
        twist = Twist()
        twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(twist)

    def publish_goal(self, x, y, yaw):
        self.goal_pub.publish(self._goal_pose(x, y, yaw))

    def _publish_string(self, publisher, text):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    def publish_status(self, text):
        self._publish_string(self.status_pub, text)

    def publish_stats(self, text):
        self._publish_string(self.stats_pub, text)

    def publish_event(self, text):
        self._publish_string(self.events_pub, text)

    def publish_map_save_status(self, text):
        self._publish_string(self.map_save_status_pub, text)

    def request_map_save(self):
        self.log(
            'info', f'Requesting map save of {self.cfg.map_save_topic}')
        self._publish_string(
            self.map_save_request_pub, self.cfg.map_save_topic)

    def point_sensor(self, bearing):
        if self.sensor_pub is None:
            return
        msg = Float32()
        msg.data = float(bearing)
        self.sensor_pub.publish(msg)

    def publish_markers(self, items):
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        stamp = self.get_clock().now().to_msg()
        frame = self._global_frame or 'map'
        for i, item in enumerate(items):
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = 'frontiers'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(item['x'])
            m.pose.position.y = float(item['y'])
            m.pose.position.z = 0.1
            m.pose.orientation.w = 1.0
            if item['best']:
                m.scale.x = m.scale.y = m.scale.z = 0.5
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.15
                m.color.r, m.color.g, m.color.b = _CLASS_COLORS.get(
                    item['cls'], (0.5, 0.5, 0.5))
            m.color.a = 1.0
            m.lifetime = Duration(sec=0)
            array.markers.append(m)
        self.frontier_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = ExplorationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:                # e.g. ExternalShutdownException
        if rclpy.ok():
            node.get_logger().error(f'spin ended: {exc}')
    finally:
        try:
            # Never leave a direct rotation command active on exit.
            node.controller.supervisor.halt_direct_motion()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()