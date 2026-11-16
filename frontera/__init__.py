"""
Modular frontier-exploration framework for ROS 2.

Requires only: a nav_msgs/OccupancyGrid map, the robot pose through TF, and a
NavigateToPose-compatible action server. Nothing in this package depends on a
specific robot, SLAM system or navigation stack; every setting comes from
exploration_manager.yaml.

Modules
  config.py                 all parameters and their defaults
  timebase.py               simulation-safe behavior clock
  grid.py                   cached occupancy grid and masks
  frontier_detection.py     frontier cells and clusters
  reachability.py           obstacle-aware path distance (one wavefront)
  information_gain.py       visibility-aware expected gain (ray casting)
  frontier_scoring.py       candidate classes and utility function
  exploration_memory.py     spatial memory of failures / visits
  navigation_supervisor.py  safe execution of one goal (transactions)
  completion.py             map progress and end-of-exploration criteria
  controller.py             decision loop tying everything together
"""

from .config import ExplorationConfig
from .controller import ExplorationController
from .grid import GridMap

__all__ = ['ExplorationConfig', 'ExplorationController', 'GridMap']
