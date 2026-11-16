"""
Frontera — supervised frontier exploration for ROS 2.

Requires only: a nav_msgs/OccupancyGrid map, the robot pose through TF, and a
NavigateToPose-compatible action server. Nothing in this package depends on a
specific robot, SLAM system or navigation stack; every setting comes from the
parameter file (config/frontera.yaml).

All modules except exploration_manager_node.py are plain Python with no ROS
dependency, so the decision logic can be tested without ROS.

Modules
  config.py                    all parameters, their defaults and validation
  timebase.py                  simulation-safe behavior clock
  grid.py                      cached occupancy grid and masks
  frontier_detection.py        frontier cells and clusters
  reachability.py              obstacle-aware path distance (one wavefront)
  information_gain.py          visibility-aware expected gain (ray casting)
  frontier_scoring.py          candidate classes and utility function
  exploration_memory.py        unified spatial exploration memory
  navigation_supervisor.py     transactional execution of one goal
  completion.py                map progress and evidence-based completion
  controller.py                decision loop tying everything together
  exploration_manager_node.py  ROS 2 node: parameters, topics, TF, action
"""

from .config import ExplorationConfig
from .controller import ExplorationController
from .grid import GridMap

__all__ = ['ExplorationConfig', 'ExplorationController', 'GridMap']
