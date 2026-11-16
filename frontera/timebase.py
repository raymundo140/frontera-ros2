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
    """Monotonic behavior time derived from ROS time.

    The controller calls update() once per decision tick and measures every
    duration (timeouts, cooldowns, budgets) as a difference of the values it
    returns.

    Attributes:
        t: Current behavior time [s]; never decreases.
        backward_jumps: Number of backward ROS-time jumps detected.
        forward_clamps: Number of forward jumps larger than
            max_forward_step_s that were clamped.
    """

    def __init__(self, max_forward_step_s=10.0, frozen_wall_timeout_s=10.0):
        """Create a clock that starts at behavior time zero.

        Args:
            max_forward_step_s: Largest ROS-time advance accepted between
                two updates [s]; larger jumps count as this much.
            frozen_wall_timeout_s: Wall time without any ROS-time advance
                after which the clock is reported as frozen [s].
        """
        self.max_forward_step_s = float(max_forward_step_s)
        self.frozen_wall_timeout_s = float(frozen_wall_timeout_s)

        self.t = 0.0                    # behavior time [s], monotonic
        self.last_ros_s = None
        self.last_advance_wall_s = None

        self.backward_jumps = 0
        self.forward_clamps = 0
        self._reset_pending = False

    def update(self, ros_now_s, wall_now_s):
        """Advance behavior time from a new pair of clock readings.

        The first call only initializes the clock. A backward ROS-time jump
        adds no time and marks a pending reset (see consume_reset()); a
        forward jump larger than max_forward_step_s is clamped.

        Args:
            ros_now_s: Current ROS time [s] (simulation time if enabled).
            wall_now_s: Current monotonic wall time [s].

        Returns:
            Behavior time after the update [s].
        """
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
        """Report whether ROS time has stopped advancing.

        Args:
            wall_now_s: Current monotonic wall time [s].

        Returns:
            True when ROS time has not advanced for longer than
            frozen_wall_timeout_s of wall time; False before the first update.
        """
        if self.last_advance_wall_s is None:
            return False
        return (
            float(wall_now_s) - self.last_advance_wall_s
            > self.frozen_wall_timeout_s
        )

    def consume_reset(self):
        """Return True once after each detected backward clock jump.

        Lets the controller react to a simulation reset exactly once (it
        clears the TF buffer, whose transforms carry old-timeline stamps).
        """
        pending = self._reset_pending
        self._reset_pending = False
        return pending
