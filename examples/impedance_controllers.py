import numpy as np
from pybullet_robot.bullet_robot import BulletRobot


# pylint: disable=C0116
def quat2rpy(quat: np.ndarray) -> tuple:
    q1, q2, q3, q0 = quat
    roll = np.arctan2(
        2 * ((q2 * q3) + (q0 * q1)), q0**2 - q1**2 - q2**2 + q3**2
    )  # radians
    pitch = np.arcsin(2 * ((q1 * q3) - (q0 * q2)))
    yaw = np.arctan2(2 * ((q1 * q2) + (q0 * q3)), q0**2 + q1**2 - q2**2 - q3**2)
    return np.array((roll, pitch, yaw))


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class CartesianImpedanceController:
    """Simplified PD control for end-effector tracking."""

    def __init__(
        self,
        robot: BulletRobot,
        kp: np.array,
        kd: np.ndarray,
        null_kp: np.ndarray,
        nullspace_pos_target: np.ndarray,
    ):

        self._robot = robot
        self._kp = kp
        self._kd = kd
        self._null_kp = null_kp
        self._nullspace_target = nullspace_pos_target
        self._goal_pos: np.ndarray = None
        self._goal_ori: np.ndarray = None

    def set_target(self, goal_pos: np.ndarray, goal_ori: np.ndarray):
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori

    def compute_cmd(self):
        curr_pos, curr_ori = self._robot.get_link_pose(
            link_id=self._robot.get_link_id(link_name=self._robot.ee_names[0])
        )
        curr_joint_pos = self._robot.get_actuated_joint_positions()
        delta_pos = self._goal_pos - curr_pos

        delta_ori = wrap_angle(
            wrap_angle(quat2rpy(self._goal_ori)) - wrap_angle(quat2rpy(curr_ori))
        )

        curr_vel, curr_omg = self._robot.get_link_velocity(
            link_id=self._robot.get_link_id(link_name=self._robot.ee_names[0])
        )
        print(wrap_angle(quat2rpy(self._goal_ori)), wrap_angle(quat2rpy(curr_ori)))

        cmd_force = np.zeros(6)
        cmd_force[:3] = self._kp[:3] * delta_pos - self._kd[:3] * curr_vel
        cmd_force[3:] = self._kp[3:] * delta_ori - self._kd[3:] * curr_omg

        error = np.asarray([np.linalg.norm(delta_pos), np.linalg.norm(delta_ori)])

        jac = self._robot.get_jacobian(ee_link_name=self._robot.ee_names[0])
        null_space_filter = self._null_kp.dot(
            np.eye(curr_joint_pos.size) - jac.T.dot(np.linalg.pinv(jac.T, rcond=1e-3))
        )
        tau = jac.T.dot(cmd_force.reshape([-1, 1])) + null_space_filter.dot(
            (self._nullspace_target - curr_joint_pos)
        )
        # joint torques to be commanded
        return tau.flatten(), error


class JointImpedanceController:
    """Simplified PD control for end-effector tracking."""

    def __init__(self, robot: BulletRobot, kp: np.array, kd: np.ndarray):

        self._robot = robot
        self._kp = kp
        self._kd = kd
        self._goal_joint_pos: np.ndarray = None

    def set_target(self, goal_joint_pos: np.ndarray):
        self._goal_joint_pos = goal_joint_pos

    def compute_cmd(self):
        curr_joint_pos = self._robot.get_actuated_joint_positions()
        curr_joint_vel = self._robot.get_actuated_joint_velocities()

        delta_pos = self._goal_joint_pos - curr_joint_pos
        # Desired joint effort commands computed using PD law
        tau = self._kp * delta_pos - self._kd * curr_joint_vel

        # joint torques to be commanded
        return tau, np.linalg.norm(delta_pos)
