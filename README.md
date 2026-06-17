# PyBullet Robot

[![PyPI version](https://badge.fury.io/py/pybullet-robot.svg)](https://badge.fury.io/py/pybullet-robot)

This package provides:
1. A generel Python interface class for robot simulations using [PyBullet](https://www.pybullet.org). Python API class to control and monitor the robot in the simulation.
2. A pybullet inverse kinematics interface that can be used for getting joint position values when end-effector/link targets are provided. Supports multi-end-effector targets and floating-base robots.


## Installation

### From PyPI

```bash
pip install pybullet_robot
```

### From source

```bash
git clone -b main https://github.com/justagist/pybullet_robot
cd pybullet_robot
pip install .
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

(TODO)
