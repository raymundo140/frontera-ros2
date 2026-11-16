"""
timebase.py — clock handling that is safe under simulation.

Two clocks are involved:

* ROS time (simulation time when use_sim_time is true). All behavior
  durations — preview, rotation deadlines, navigation budgets, cooldowns,
  progress windows — must be measured with it, otherwise they change with
  the simulator's real-time factor.
* Monotonic wall time. Used only to notice that ROS time stopped advancing
  and as an emergency bound for components that do not answer.

BehaviorClock turns raw ROS time into a "behavior time" that never goes
backward: a backward jump or reset of the simulation clock contributes zero
elapsed time, and an implausibly large forward jump is clamped. Every
deadline in the framework is expressed in behavior time, so no duration can
become negative or stale after a clock reset.
"""


class BehaviorClock:

    def __init__(self, max_forward_step_s=10.0, frozen_wall_timeout_s=10.0):
        self.max_forward_step_s = float(max_forward_step_s)
        self.frozen_wall_timeout_s = float(frozen_wall_timeout_s)

        self.t = 0.0                    # behavior time [s], monotonic
        self.last_ros_s = None
        self.last_advance_wall_s = None

        self.backward_jumps = 0
        self.forward_clamps = 0
        self._reset_pending = False

    def update(self, ros_now_s, wall_now_s):
        """Feed the current ROS and wall time; return behavior time."""
        ros_now_s = float(ros_now_s)
        wall_now_s = float(wall_now_s)

        if self.last_ros_s is None:
            self.last_ros_s = ros_now_s
            self.last_advance_wall_s = wall_now_s
            return self.t

        delta = ros_now_s - self.last_ros_s

        if delta < 0.0:
            # Simulation reset / backward jump: no behavior time elapses.
            self.backward_jumps += 1
            self._reset_pending = True
            delta = 0.0
            self.last_advance_wall_s = wall_now_s
        elif delta > self.max_forward_step_s:
            self.forward_clamps += 1
            delta = self.max_forward_step_s

        if delta > 0.0:
            self.last_advance_wall_s = wall_now_s

        self.t += delta
        self.last_ros_s = ros_now_s
        return self.t

    def is_frozen(self, wall_now_s):
        """True when ROS time has not advanced for frozen_wall_timeout_s."""
        if self.last_advance_wall_s is None:
            return False
        return (
            float(wall_now_s) - self.last_advance_wall_s
            > self.frozen_wall_timeout_s
        )

    def consume_reset(self):
        """Return True once after each detected backward clock jump."""
        pending = self._reset_pending
        self._reset_pending = False
        return pending
