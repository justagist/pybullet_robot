"""Joint-space position control + robot introspection demo.

Loads a KUKA iiwa14, prints the robot's structure (joints, links, masses, limits) using the
built-in verbose info, then drives the joints along a smooth sinusoidal trajectory with
position control. The live robot state from ``get_robot_states`` is printed periodically.

This is the simplest "getting started" demo: load a robot, inspect it, command joints, read
state.

Run:
    python demo_joint_position_control.py
"""

import time

import numpy as np

from pybullet_robot.bullet_robot import BulletRobot
from pybullet_robot.utils.robot_loader_utils import (
    get_urdf_from_awesome_robot_descriptions,
)


def main():
    robot = BulletRobot(
        urdf_path=get_urdf_from_awesome_robot_descriptions("iiwa14_description"),
        ee_names=["iiwa_link_ee"],
        run_async=False,  # step the simulation manually
        place_on_ground=False,
        verbose=True,  # prints the full robot info table on construction
    )
    robot.set_position_control_mode()

    n = robot.num_actuated_joints
    home = robot.get_actuated_joint_positions().copy()
    # per-joint sinusoid amplitudes (radians); trimmed to the robot's DOF
    amplitude = np.deg2rad([30, 25, 40, 35, 45, 40, 60])[:n]
    frequency = 0.2  # Hz

    print(
        "\nRunning a sinusoidal joint trajectory with position control. Ctrl+C to exit.\n"
    )
    start = time.time()
    last_log = 0.0
    try:
        while True:
            t = time.time() - start
            target = home + amplitude * np.sin(2 * np.pi * frequency * t)
            robot.set_joint_positions(cmd=target)
            robot.step()

            now = time.time()
            if now - last_log > 1.0:
                state = robot.get_robot_states()
                ee_pos = robot.get_link_pose(robot.ee_ids[0])[0]
                tracking_err = np.linalg.norm(
                    target - state["actuated_joint_positions"]
                )
                print(
                    f"t={t:5.1f}s | ee_pos={np.round(ee_pos, 3)} "
                    f"| joint tracking err={tracking_err:.4f} rad"
                )
                last_log = now
            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
