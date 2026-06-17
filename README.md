# PyBullet Robot

[![PyPI version](https://badge.fury.io/py/pybullet-robot.svg)](https://badge.fury.io/py/pybullet-robot)

This package provides:

1. [`BulletRobot`]: A general Python interface class for robot simulations using [PyBullet](https://www.pybullet.org). Python API class to control and monitor the robot in the simulation.
2. [`PybulletIKInterface`] class: A constraint-based inverse kinematics (IK) "solver" interface. Instead of solving IK analytically, it attaches PyBullet constraints that pull the requested end-effector/link frames towards their target poses, forward-simulates the physics until they settle, and reads back the resulting joint configuration as the IK solution. It supports tracking several frames at once (full 6-DOF pose targets and position-only targets), and floating-base robots (where the base pose is solved together with the joint angles).

## Installation

### From PyPI

```bash
pip install pybullet_robot
```

### From source

```bash
pip install git+https://github.com/justagist/pyrcf
```

## Development

Install in editable mode with the development extras (test + lint + build tooling):

```bash
git clone -b main https://github.com/justagist/pybullet_robot
cd pybullet_robot
pip install -e ".[dev]"
```

A [dev container](.devcontainer/devcontainer.json) is provided for a reproducible
environment in VS Code ("Reopen in Container").

## Usage

See [examples/](examples/) for runnable demos covering different robots and capabilities:

- **[Joint position control](examples/demo_joint_position_control.py)** (KUKA iiwa14): loading a
  robot, inspecting it, position control, reading state.
- **[Task-space control](examples/demo_task_space_control.py)** (Franka Panda): Cartesian
  impedance / torque control with gravity compensation and nullspace posture control.
- **[Inverse kinematics](examples/demo_ik_interface.py)** (KUKA iiwa14): interactive IK; drag
  GUI sliders to set an end-effector target and the solver reaches it.
- **[Headless IK + separate-sim control](examples/demo_ik_headless_control.py)** (KUKA iiwa14):
  run the IK interface headless as a pure solver and apply its solution to a robot controlled in
  a separate pybullet GUI instance.
- **[Whole-body / multi-end-effector IK](examples/demo_quadruped_ik.py)** (Unitree A1):
  floating-base IK tracking the body pose and all four feet simultaneously.

```bash
cd examples
python demo_ik_interface.py
```

Robot models are downloaded automatically on first run via `robot_descriptions`.
