# PyBullet Robot

[![PyPI version](https://img.shields.io/pypi/v/pybullet-robot)](https://pypi.org/project/pybullet-robot/)
[![CI](https://github.com/justagist/pybullet_robot/actions/workflows/ci.yml/badge.svg)](https://github.com/justagist/pybullet_robot/actions/workflows/ci.yml)

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

See [examples/](examples/) for runnable demos covering different robots and capabilities. Robot
models are downloaded automatically on first run via `robot_descriptions`:

```bash
cd examples
python demo_task_space_control.py
```

## Examples

### Joint position control

Loading a robot, inspecting it, position control, and reading state.
([`demo_joint_position_control.py`](examples/demo_joint_position_control.py))

![Joint position control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_joint_position_control.gif)

### Task-space control

Example using a custom cartesian impedance / torque control for task-space control.
([`demo_task_space_control.py`](examples/demo_task_space_control.py))

![Task-space control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_task_space_control.gif)

### Headless IK + separate-sim control

Run the provided `PybulletIKInterface` headless as a pure solver and apply its solution to a robot controlled via position control.
([`demo_ik_headless_control.py`](examples/demo_ik_headless_control.py))

![Headless IK control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_ik_headless_control.gif)

### Whole-body / multi-end-effector IK

Floating-base IK tracking the body pose and all four feet simultaneously.
([`demo_quadruped_ik.py`](examples/demo_quadruped_ik.py))

![Quadruped whole-body IK demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_quadruped_ik.gif)

### Interactive inverse kinematics (KUKA iiwa14)

Interactive IK: drag GUI sliders to set an end-effector target and the constraint-based solver
reaches it. ([`demo_ik_interface.py`](examples/demo_ik_interface.py)) Visually similar to the
headless IK demo above, so it has no separate clip; run it to try it interactively.
