# Frontera

**Supervised frontier exploration for ROS 2.**


Frontera drives a mobile robot through an unknown indoor environment until the
map is complete. It decides where the robot should go next, supervises every
navigation goal until it ends, and is built on five mechanisms: transactional
goal execution, path- and information-aware selection, progress-aware
supervision, unified spatial exploration memory, and evidence-based
completion.

It is a single ROS 2 node. It needs no specific robot, SLAM system, or
navigation stack, only three standard interfaces:

1. a `nav_msgs/OccupancyGrid` map (for example from SLAM Toolbox),
2. the robot pose in TF (`map -> base`),
3. a `nav2_msgs/action/NavigateToPose` action server (for example Nav2).

## Mechanisms

| Mechanism | What it does |
|---|---|
| Transactional goal execution | One decision gate selects goals; each goal is a transaction with an ID, so late navigation callbacks cannot affect a newer goal. A goal ends only when the navigation stack confirms it. |
| Path- and information-aware selection | Candidates are ranked by expected information gain, path length through one wavefront, heading continuity, a penalty for revisiting recent goals, and the record of their region. |
| Progress-aware supervision | Each goal receives a time budget scaled by its path length, may be extended once while the robot advances, and is canceled when the robot stalls. |
| Unified spatial exploration memory | Outcomes are stored per region rather than per grid cell: failures impose growing cooldowns and finally exhaust a region, unproductive visits lower its value, and new map near an exhausted region reopens it once. |
| Evidence-based completion | Exploration ends as `COMPLETE` or `COMPLETE_PARTIAL` only after several consecutive maps confirm that no eligible frontier remains; `STAGNATED` and `TIME_LIMIT` are reported separately. |

All decision logic is plain Python with no ROS dependency (`frontera/`); the
node (`frontera/exploration_manager_node.py`) only connects it to ROS.

## Requirements

- ROS 2 Humble (other distributions untested)
- Python 3 with NumPy; SciPy is optional and makes some steps faster
- A SLAM node, TF, and a `NavigateToPose` server, as listed above

## Installation

```bash
cd ~/ros2_ws/src
git clone https://github.com/raymundo140/frontera-ros2.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -y
colcon build --packages-select frontera
source install/setup.bash
```

## Usage

Start your robot, SLAM, and navigation stack first, then:

```bash
ros2 launch frontera frontera.launch.py params_file:=/path/to/your.yaml
# in simulation, add: use_sim_time:=true
```

The node waits and never moves the robot until it receives `START`:

```bash
ros2 topic pub --once /exploration/control std_msgs/msg/String "{data: START}"
```

Other commands on the same topic: `PAUSE`, `STOP`, `SAVE_MAP`,
`RESET_NAV_FAULT`.

## Configuration

Every setting is a ROS 2 parameter, documented line by line in
[`config/frontera.yaml`](config/frontera.yaml). Section 1 of that file is
robot-specific and must be reviewed for a new robot: map topic, frames,
velocity topic, navigation speed, sensor range, and the optional
rotate-first behavior. The remaining sections hold behavior settings with
conservative starting values.

[`config/call_m.yaml`](config/call_m.yaml) is the complete configuration used
on the CALL-M mobile manipulator, kept as a worked example.

## Interface

| Topic | Type | Direction | Content |
|---|---|---|---|
| `/exploration/control` | `std_msgs/String` | in | `START`, `PAUSE`, `STOP`, `SAVE_MAP`, `RESET_NAV_FAULT` |
| `/exploration/status` | `std_msgs/String` | out | current state, 1 Hz |
| `/exploration/stats` | `std_msgs/String` | out | key=value statistics, 1 Hz |
| `/exploration/events` | `std_msgs/String` | out | one JSON object per decision or outcome |
| `/exploration/goal` | `geometry_msgs/PoseStamped` | out | current target |
| `/exploration/frontiers` | `visualization_msgs/MarkerArray` | out | candidates, for RViz |
| `/exploration/map_save_request` | `std_msgs/String` | out | asks your launcher to save the map |

Frontera never writes map files itself. At the end of exploration it publishes
a save request, so that the process that owns your system's lifecycle saves
the map.

## Tests

The decision logic is tested without ROS, using a fake clock, TF, and action
server:

```bash
cd frontera-ros2
python3 -m pytest test/ -v
```

## Status

Frontera was developed and evaluated on the CALL-M mobile manipulator and in
Gazebo simulation with SLAM Toolbox and Nav2 under ROS 2 Humble. Other robots
are expected to need only the configuration in section 1 of the parameter
file, but have not been tested.

## Citation

Frontera accompanies the paper *A Supervised ROS 2 Architecture for Autonomous
Mapping: Modular Exploration-Management Framework*, IEEE/SICE International
Symposium on System Integration (SII), 2027. If you use it in your work,
please cite that paper.

## License

MIT. See [LICENSE](LICENSE).
