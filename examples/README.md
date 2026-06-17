# Examples

Runnable demos showcasing the package across different robots and capabilities. Robot models
are fetched automatically on first run via
[`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py) (cached
afterwards), so the first launch of each demo may take a moment to download.

Run them from this directory:

```bash
cd examples
python demo_task_space_control.py
```

| Demo                                                             | Robot        | Demonstrates                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [demo_joint_position_control.py](demo_joint_position_control.py) | KUKA iiwa14  | Loading a robot, printing its structure, joint-space **position control**, reading robot state.                                                                                                                                                                             |
| [demo_task_space_control.py](demo_task_space_control.py)         | Franka Panda | **Cartesian impedance / torque control** -- the end-effector tracks a circle while the arm holds itself up (gravity compensation) and regulates its posture in the nullspace.                                                                                               |
| [demo_ik_interface.py](demo_ik_interface.py)                     | KUKA iiwa14  | **Inverse kinematics** via `PybulletIKInterface`. Drag the GUI sliders to set a target end-effector pose; the IK solver finds joint angles to reach it.                                                                                                                     |
| [demo_ik_headless_control.py](demo_ik_headless_control.py)       | KUKA iiwa14  | **Headless IK + control in a separate sim**. The IK interface runs headless (DIRECT) as a pure solver; its joint solution is applied to a robot in a *separate* pybullet GUI instance via position control. This is the typical intended usecase for `PybulletIKInterface`. |
| [demo_quadruped_ik.py](demo_quadruped_ik.py)                     | Unitree A1   | **Multi-end-effector, floating-base IK**. Body pose + four feet are tracked simultaneously; sliders can be used to move the body trunk while the feet stay planted and the 12 leg joints (and the free base) are solved.                                                    |

`impedance_controllers.py` holds the `CartesianImpedanceController` used by the torque-control demos; `demo_utils.py` holds small cosmetic helpers (clean solid ground plane, camera/lighting) shared by the demos.

## Previews

### Joint position control (KUKA iiwa14)

![Joint position control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_joint_position_control.gif)

### Task-space control (Franka Panda)

![Task-space control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_task_space_control.gif)

### Headless IK + separate-sim control (KUKA iiwa14)

![Headless IK control demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_ik_headless_control.gif)

### Whole-body / multi-end-effector IK (Unitree A1)

![Quadruped whole-body IK demo](https://media.githubusercontent.com/media/justagist/_assets/refs/heads/main/pybullet_robot/demo_quadruped_ik.gif)

(`demo_ik_interface.py` is interactive and visually similar to the headless IK demo above, so it
has no separate clip.)

## Notes

- The impedance/torque demos use `enable_torque_mode=True` and add
  `robot.get_gravity_compensation_torques()` so the arm does not sag under gravity.
- The IK demos open a pybullet GUI with sliders. The IK interface "solves" by forward-simulating
  physics constraints, so it converges to a target once set rather than tracking a fast-moving
  goal -- set a target with the sliders and watch it solve.
