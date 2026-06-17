"""Task-space (Cartesian) control demo.

Loads a Franka Panda in torque/impedance mode and uses a Cartesian impedance controller to
make the end-effector trace a circle in the world Y-Z plane, demonstrating end-effector
task-space tracking while the arm holds itself up (gravity compensation) and regulates its
posture in the nullspace.
"""

import time
import numpy as np
from pybullet_robot.bullet_robot import BulletRobot
from pybullet_robot.utils.robot_loader_utils import (
    get_urdf_from_awesome_robot_descriptions,
)

from impedance_controllers import CartesianImpedanceController

NEUTRAL_JOINT_POS = [
    -0.017792060227770554,
    -0.7601235411041661,
    0.019782607023391807,
    -2.342050140544315,
    0.029840531355804868,
    1.5411935298621688,
    0.7534486589746342,
    0.0,
    0.0,
]

# Cartesian impedance gains: [x, y, z, roll, pitch, yaw].
# Gravity compensation is handled by the controller, so moderate (compliant) stiffness is
# enough for smooth, stable tracking.
Kp = np.array([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
Kd = 2.0 * np.sqrt(Kp)  # critically damped
# Gentle nullspace stiffness to keep the elbow near the neutral posture.
NULLSPACE_Kp = np.array([10.0] * 9)

# Circular reference trajectory (in the world Y-Z plane).
TRAJ_RADIUS = 0.12  # meters
TRAJ_PERIOD = 6.0  # seconds per revolution

if __name__ == "__main__":
    robot = BulletRobot(
        urdf_path=get_urdf_from_awesome_robot_descriptions("panda_description"),
        default_joint_positions=NEUTRAL_JOINT_POS,
        enable_torque_mode=True,  # enable to be able to use torque control
        ee_names=["panda_hand"],
        run_async=False,  # set to False to enable stepping simulation manually
    )

    controller = CartesianImpedanceController(
        robot=robot,
        kp=Kp,
        kd=Kd,
        null_kp=NULLSPACE_Kp,
        nullspace_pos_target=NEUTRAL_JOINT_POS,
    )

    robot.step()
    center_pos, fixed_ori = robot.get_link_pose(link_id=robot.ee_ids[0])
    center_pos = center_pos.copy()
    fixed_ori = fixed_ori.copy()

    omega = 2.0 * np.pi / TRAJ_PERIOD
    start_time = time.time()
    n_iters = 0
    while True:
        t = time.time() - start_time
        # circle in the Y-Z plane, starting smoothly from the initial end-effector position
        goal_pos = center_pos + np.array(
            [
                0.0,
                TRAJ_RADIUS * np.sin(omega * t),
                TRAJ_RADIUS * (1.0 - np.cos(omega * t)),
            ]
        )
        controller.set_target(goal_pos=goal_pos, goal_ori=fixed_ori)

        joint_cmds, error = controller.compute_cmd()
        robot.set_actuated_joint_commands(tau=joint_cmds)

        robot.step()  # step simulation
        if n_iters % 50 == 0:
            print(f"pos error: {error[0]:.4f} m | ori error: {error[1]:.4f} rad")
        n_iters += 1
        time.sleep(0.01)
