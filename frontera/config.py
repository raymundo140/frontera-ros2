"""
config.py — every tunable value of the exploration framework.

The dataclass below is the single list of parameters. The ROS node declares
one ROS 2 parameter per field (same name), so everything here can be set from
the external exploration_manager.yaml without touching code.

The defaults are robot-agnostic starting values. They are NOT experimentally
optimized; the documentation for each parameter lives in the YAML file.
"""

from dataclasses import dataclass, field, fields
from typing import List


@dataclass
class ExplorationConfig:
    # ── ROS interfaces ───────────────────────────────────────────────────
    map_topic: str = '/map'
    map_qos_transient_local: bool = True
    costmap_topic: str = '/global_costmap/costmap'
    odom_topic: str = ''
    cmd_vel_topic: str = '/cmd_vel'
    navigate_action: str = '/navigate_to_pose'
    control_topic: str = '/exploration/control'
    status_topic: str = '/exploration/status'
    stats_topic: str = '/exploration/stats'
    goal_topic: str = '/exploration/goal'
    frontiers_topic: str = '/exploration/frontiers'
    events_topic: str = '/exploration/events'
    map_save_request_topic: str = '/exploration/map_save_request'
    map_save_status_topic: str = '/exploration/map_save_status'
    sensor_look_topic: str = ''
    map_save_topic: str = '/map'
    request_map_save_on_complete: bool = True
    legacy_complete_status: bool = True

    # ── frames / TF ──────────────────────────────────────────────────────
    global_frame: str = ''
    robot_base_frame: str = 'base_link'
    odom_frame: str = ''
    tf_map_odom_max_age_s: float = 0.0
    robot_base_frame_fallbacks: List[str] = field(
        default_factory=lambda: ['base_footprint', 'base_link'])
    use_odom_child_frame: bool = True
    tf_max_age_s: float = 1.0
    tf_future_tolerance_s: float = 5.0
    tf_accept_zero_stamp: bool = True

    # ── clock safety ─────────────────────────────────────────────────────
    clock_max_forward_step_s: float = 10.0
    frozen_clock_wall_timeout_s: float = 10.0

    # ── loop rates ───────────────────────────────────────────────────────
    decision_rate_hz: float = 5.0
    rotation_control_rate_hz: float = 20.0
    status_rate_hz: float = 1.0

    # ── occupancy-grid interpretation ────────────────────────────────────
    map_free_max_value: int = 25
    map_occupied_min_value: int = 50
    costmap_reject_cost: int = 80

    # ── robot / sensor model ─────────────────────────────────────────────
    expected_speed_mps: float = 0.25
    sensor_range_m: float = 3.0
    sensor_fov_deg: float = 360.0

    # ── frontier detection ───────────────────────────────────────────────
    min_frontier_length_m: float = 0.15
    max_frontier_length_m: float = 3.0
    map_edge_margin_m: float = 0.10

    # ── goal validity ────────────────────────────────────────────────────
    goal_clearance_m: float = 0.25
    min_goal_distance_m: float = 0.7
    goal_reached_distance_m: float = 0.55

    # ── reachability / path distance ─────────────────────────────────────
    reachability_enabled: bool = True
    reachability_inflation_m: float = 0.20
    reachability_resolution_m: float = 0.10
    reachability_start_search_m: float = 0.60
    unreachable_distance_factor: float = 2.0
    unreachable_confirm_maps: int = 2
    probe_unreachable_with_nav: bool = True
    unreachable_probe_limit: int = 1

    # ── information gain ─────────────────────────────────────────────────
    gain_num_rays: int = 72
    gain_ray_step_m: float = 0.05
    min_information_gain_m2: float = 0.10

    # ── utility  U = Gn * exp(-lambda * L) * C * R * V ───────────────────
    travel_cost_lambda: float = 0.3
    direction_continuity_weight: float = 0.45
    direction_min_update_distance_m: float = 0.40
    direction_memory_keep: float = 0.65
    direction_lookahead_m: float = 1.0
    recent_goal_history_size: int = 6
    recent_goal_radius_m: float = 1.0
    recent_goal_penalty: float = 0.35
    unproductive_visit_penalty: float = 0.5

    # ── spatial exploration memory ───────────────────────────────────────
    region_match_radius_m: float = 0.75
    # Controlled experimental policies; defaults reproduce the original v2.
    # 'exact_goal' changes navigation-failure lookup ONLY; reachability and
    # unproductive visits retain their normal spatial-region memory.
    failure_memory_policy: str = 'region'          # region | exact_goal
    failure_goal_quantization_m: float = 0.05      # world-coordinate bins
    failure_recovery_policy: str = 'progressive'    # progressive | fixed
    fixed_failure_cooldown_s: float = 30.0
    failure_cooldowns_s: List[float] = field(
        default_factory=lambda: [30.0, 120.0])
    exhaust_after_failures: int = 3
    unproductive_exhaust_after: int = 2
    min_productive_gain_m2: float = 0.25
    productivity_radius_m: float = 1.5
    settle_maps: int = 1
    settle_timeout_s: float = 3.0
    reopen_radius_m: float = 1.5
    reopen_min_new_area_m2: float = 0.5
    max_reopens_per_region: int = 1

    # ── goal execution: preview + rotate-first ───────────────────────────
    goal_preview_s: float = 3.0
    rotate_first_enabled: bool = True
    rotation_heading_offset_deg: float = 0.0
    rotation_start_tolerance_rad: float = 0.20
    rotation_done_tolerance_rad: float = 0.15
    rotation_kp: float = 1.0
    rotation_max_angular_vel: float = 0.5
    rotation_min_angular_vel: float = 0.12
    rotation_timeout_s: float = 30.0
    rotation_stall_timeout_s: float = 6.0
    rotation_min_progress_rad: float = 0.05
    rotation_tf_loss_timeout_s: float = 2.0
    rotation_required_clearance_m: float = 0.0

    # ── navigation supervision ───────────────────────────────────────────
    goal_response_timeout_s: float = 10.0
    goal_response_wall_timeout_s: float = 30.0
    nav_budget_base_s: float = 30.0
    nav_budget_factor: float = 3.0
    nav_budget_min_s: float = 60.0
    nav_budget_max_s: float = 600.0
    nav_budget_extensions_max: int = 1
    progress_timeout_s: float = 60.0
    progress_min_improvement_m: float = 0.25
    progress_recovery_graces: int = 3
    resolved_confirm_maps: int = 2
    resolved_gain_fraction: float = 0.15
    cancel_timeout_s: float = 10.0
    cancel_wall_timeout_s: float = 60.0
    cancel_retries: int = 1
    nav_fault_abort_timeout_s: float = 120.0

    # ── completion ───────────────────────────────────────────────────────
    completion_policy: str = 'evidence'  # evidence | single_no_eligible
    completion_confirm_maps: int = 3
    idle_reevaluation_period_s: float = 5.0
    stagnation_timeout_s: float = 180.0
    stagnation_min_growth_m2: float = 0.5
    map_progress_window_s: float = 60.0
    max_exploration_time_s: float = 0.0
    map_stable_change_threshold_pct: float = 0.1

    # ── instrumentation ──────────────────────────────────────────────────
    event_max_candidates: int = 10
    pose_jump_reject_m: float = 1.0
    log_decisions: bool = True

    # ---------------------------------------------------------------------
    def validate(self):
        """Raise ValueError for values that cannot work; return warnings."""
        warnings = []

        def positive(name):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f'{name} must be > 0')

        def non_negative(name):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f'{name} must be >= 0')

        for name in (
                'decision_rate_hz', 'rotation_control_rate_hz',
                'status_rate_hz', 'expected_speed_mps', 'sensor_range_m',
                'sensor_fov_deg', 'goal_reached_distance_m',
                'region_match_radius_m', 'rotation_max_angular_vel',
                'rotation_timeout_s', 'nav_budget_min_s', 'nav_budget_max_s',
                'progress_timeout_s', 'cancel_timeout_s',
                'cancel_wall_timeout_s', 'goal_response_timeout_s',
                'goal_response_wall_timeout_s', 'tf_max_age_s',
                'clock_max_forward_step_s', 'frozen_clock_wall_timeout_s',
                'stagnation_timeout_s', 'map_progress_window_s',
                'productivity_radius_m', 'reopen_radius_m'):
            positive(name)

        for name in (
                'goal_preview_s', 'min_frontier_length_m',
                'max_frontier_length_m', 'map_edge_margin_m',
                'goal_clearance_m', 'min_goal_distance_m',
                'reachability_inflation_m', 'min_information_gain_m2',
                'travel_cost_lambda', 'settle_timeout_s',
                'max_exploration_time_s', 'nav_fault_abort_timeout_s',
                'rotation_required_clearance_m', 'rotation_min_angular_vel',
                'idle_reevaluation_period_s'):
            non_negative(name)

        if not 0.0 <= self.direction_continuity_weight < 1.0:
            raise ValueError('direction_continuity_weight must be in [0, 1)')
        for name in ('recent_goal_penalty', 'unproductive_visit_penalty'):
            v = float(getattr(self, name))
            if not 0.0 < v <= 1.0:
                raise ValueError(f'{name} must be in (0, 1]')
        if not 0.0 <= self.direction_memory_keep < 1.0:
            raise ValueError('direction_memory_keep must be in [0, 1)')
        if self.sensor_fov_deg > 360.0:
            raise ValueError('sensor_fov_deg must be <= 360')
        if self.gain_num_rays < 8:
            raise ValueError('gain_num_rays must be >= 8')
        if self.exhaust_after_failures < 1:
            raise ValueError('exhaust_after_failures must be >= 1')
        if self.unproductive_exhaust_after < 1:
            raise ValueError('unproductive_exhaust_after must be >= 1')
        if self.completion_confirm_maps < 1:
            raise ValueError('completion_confirm_maps must be >= 1')
        if not self.failure_cooldowns_s:
            raise ValueError('failure_cooldowns_s must not be empty')
        if self.nav_budget_min_s > self.nav_budget_max_s:
            raise ValueError('nav_budget_min_s must be <= nav_budget_max_s')
        if self.map_free_max_value >= self.map_occupied_min_value:
            raise ValueError(
                'map_free_max_value must be < map_occupied_min_value')

        if self.failure_memory_policy not in ('region', 'exact_goal'):
            raise ValueError('failure_memory_policy must be region or exact_goal')
        if self.failure_recovery_policy not in ('progressive', 'fixed'):
            raise ValueError('failure_recovery_policy must be progressive or fixed')
        if self.completion_policy not in ('evidence', 'single_no_eligible'):
            raise ValueError('completion_policy must be evidence or single_no_eligible')
        positive('failure_goal_quantization_m')
        non_negative('fixed_failure_cooldown_s')

        if self.reachability_inflation_m > self.goal_clearance_m:
            warnings.append(
                'reachability_inflation_m > goal_clearance_m: a valid goal '
                'could look unreachable; clamping reachability_inflation_m '
                'to goal_clearance_m')
            self.reachability_inflation_m = self.goal_clearance_m
        if self.min_goal_distance_m < self.goal_reached_distance_m:
            warnings.append(
                'min_goal_distance_m < goal_reached_distance_m: goals may '
                'be counted as reached without any motion')
        if self.rotation_done_tolerance_rad > self.rotation_start_tolerance_rad:
            warnings.append(
                'rotation_done_tolerance_rad > rotation_start_tolerance_rad')
        return warnings

    @classmethod
    def field_defaults(cls):
        """Return {name: default} for every parameter (used by the node)."""
        default = cls()
        return {f.name: getattr(default, f.name) for f in fields(cls)}
