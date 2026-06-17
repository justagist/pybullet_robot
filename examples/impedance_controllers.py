import numpy as np
from scipy.spatial.transform import Rotation
from pybullet_robot.bullet_robot import BulletRobot


# pylint: disable=C0116
def orientation_error(goal_quat: np.ndarray, curr_quat: np.ndarray) -> np.ndarray:
    """Orientation error as an axis-angle (rotation) vector in the world frame.

    This is the SO(3) error that is consistent with the angular-velocity rows of the geometric
    Jacobian, so it can be used directly as a moment in a Cartesian impedance law. (A naive
    difference of Euler/RPY angles is NOT consistent and leads to instability.)
    """
    return (
        Rotation.from_quat(goal_quat) * Rotation.from_quat(curr_quat).inv()
    ).as_rotvec()


class CartesianImpedanceController:
    """Simplified PD control for end-effector tracking."""

    def __init__(
        self,
        robot: BulletRobot,
        kp: np.array,
        kd: np.ndarray,
        null_kp: np.ndarray,
        nullspace_pos_target: np.ndarray,
        joint_damping: float = 1.0,
    ):

        self._robot = robot
        self._kp = np.asarray(kp, dtype=float)
        self._kd = np.asarray(kd, dtype=float)
        self._null_kp = np.asarray(null_kp, dtype=float)
        self._nullspace_target = np.asarray(nullspace_pos_target, dtype=float)
        # Uniform joint-space damping. Damping is applied in joint space (not as a Cartesian
        # wrench) because the effective rotational inertia about the end-effector axes is tiny;
        # explicit Cartesian angular damping is discretely unstable on those axes, whereas
        # joint-space damping is governed by the (larger) mass-matrix diagonal and stays stable.
        self._joint_damping = joint_damping
        self._goal_pos: np.ndarray = None
        self._goal_ori: np.ndarray = None

    def set_target(self, goal_pos: np.ndarray, goal_ori: np.ndarray):
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori

    def compute_cmd(self):
        ee_id = self._robot.get_link_id(link_name=self._robot.ee_names[0])
        curr_pos, curr_ori = self._robot.get_link_pose(link_id=ee_id)
        curr_joint_pos = self._robot.get_actuated_joint_positions()
        curr_joint_vel = self._robot.get_actuated_joint_velocities()

        jac = self._robot.get_jacobian(ee_link_name=self._robot.ee_names[0])
        # End-effector twist from the SAME Jacobian used for control (keeps damping consistent).
        ee_twist = jac.dot(curr_joint_vel)
        curr_vel = ee_twist[:3]

        delta_pos = self._goal_pos - curr_pos
        delta_ori = orientation_error(self._goal_ori, curr_ori)

        # Cartesian impedance wrench: PD on position, stiffness on orientation. Orientation
        # damping is handled by the joint-space damping term below (see __init__).
        cmd_force = np.zeros(6)
        cmd_force[:3] = self._kp[:3] * delta_pos - self._kd[:3] * curr_vel
        cmd_force[3:] = self._kp[3:] * delta_ori

        error = np.asarray([np.linalg.norm(delta_pos), np.linalg.norm(delta_ori)])

        # Nullspace projector: drive joints toward the neutral posture without disturbing the
        # end-effector task. N = I - J^T (J^T)^+
        null_proj = np.eye(curr_joint_pos.size) - jac.T.dot(
            np.linalg.pinv(jac.T, rcond=1e-3)
        )
        null_torque = null_proj.dot(
            self._null_kp * (self._nullspace_target - curr_joint_pos)
        )

        # task torque + nullspace posture + gravity compensation + joint-space damping
        tau = (
            jac.T.dot(cmd_force)
            + null_torque
            + self._robot.get_gravity_compensation_torques()
            - self._joint_damping * curr_joint_vel
        )
        # joint torques to be commanded
        return tau, error
