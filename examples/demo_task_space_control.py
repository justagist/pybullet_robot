import time
import numpy as np
from pybullet_robot.bullet_robot import BulletRobot
from pybullet_robot.utils.urdf_utils import get_urdf_from_awesome_robot_descriptions

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

Kp = np.array([12000.0, 2000.0, 2000.0, 10, 10, 10])
Kd = 2 * np.sqrt(Kp)
Kd[3:] = np.asarray([0.00, 0.00, 0.00])
NULLSPACE_Kp = np.array([1000] * 9)

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
    curr_ee_pos, curr_ee_ori = robot.get_link_pose(link_id=robot.ee_ids[0])
    controller.set_target(goal_pos=curr_ee_pos.copy(), goal_ori=curr_ee_ori.copy())

    while True:

        joint_cmds, _ = controller.compute_cmd()
        robot.set_actuated_joint_commands(tau=joint_cmds)

        robot.step()  # step simulation
        time.sleep(0.01)
