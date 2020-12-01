#! /bin/bash

# Run `source set_env.sh` if you want to add PyBullet ROBOT to python path.
# This is not required if you are using this as a catkin package in ROS workspace.

PB_ROBOT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PB_ROBOT_PATH:$PYTHONPATH

echo -e "\nAdded PyBullet ROBOT to python path. Access using `import pybullet_robot` from python env.\n"
