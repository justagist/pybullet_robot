"""Pybullet robot interface. This is a stand-alone file."""

from typing import List, Dict, Any, Tuple, TypeAlias, Mapping, Final
from dataclasses import dataclass
from numbers import Number
import pybullet as pb
import pybullet_data
import numpy as np

# pylint: disable=I1101, R0914, C0302, R0902, R0904, R0913, C0103, C0116

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap the provided angle(s) (radians) to be within -pi and pi.

    Args:
        angle (float | np.ndarray): Input angle (or array of angles) in radians.

    Returns:
        float | np.ndarray: Output after wrapping.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def fix_pb_link_mass_inertia(
    body_id: int,
    cid: int = 0,
    mass: float = 1e-9,
    local_inertia_diagonal: float = (1e-9, 1e-9, 1e-9),
):
    """Fix pybullet virtual links.

    When loading urdf, pybullet assigned default value for link with no inertial (default value
    mass=1, localinertiadiagonal = 1,1,1; identity local inertial frame). This function fixes
    the link inertial values to the given mass and local_inertia_diagonal; small values by default.

    Args:
        body_id (int): Pybullet object ID
        cid (int, optional): Pybullet physics client ID. Defaults to 0.
        mass (float, optional): Virtual mass to assign to virtual link. Defaults to 1e-9.
        local_inertia_diagonal (float, optional): Values to assign to inertia matrix diagonals for
            virtual links. Defaults to (1e-9, 1e-9, 1e-9).
    """
    link_ids = [-1] + [-1] + list(range(pb.getNumJoints(body_id, physicsClientId=cid)))
    for link_id in link_ids:
        link_info = pb.getDynamicsInfo(body_id, link_id, physicsClientId=cid)
        if link_info[0] == 1.0 and link_info[2] == (1.0, 1.0, 1.0):
            pb.changeDynamics(
                bodyUniqueId=body_id,
                linkIndex=link_id,
                mass=mass,
                localInertiaDiagonal=local_inertia_diagonal,
                physicsClientId=cid,
            )


@dataclass
class BulletJointInfo:
    """Container holding all static information from pybullet's getJointInfo for a single joint."""

    joint_id: Final[int]
    """Joint id. Use for getting joint info from pybullet api."""
    joint_name: Final[str]
    joint_type: Final[int]
    q_index: Final[int]
    """The first position index of the joint in the positional state variables for this body."""
    u_index: Final[int]
    """The first velocity index of the joint in the positional state variables for this body."""
    joint_damping: Final[float]
    joint_friction: Final[float]
    joint_lower_limit: Final[float]
    joint_upper_limit: Final[float]
    joint_max_force: Final[float]
    joint_max_velocity: Final[float]
    link_name: Final[str]
    """Child link attached to this joint."""
    joint_axis: Final[Tuple[float, float, float]]
    """NOTE: In local frame."""
    parent_frame_pos: Final[Vector3D]
    """Joint position in parent frame."""
    parent_frame_ori: Final[QuatType]
    """Joint orientation in parent frame. Quaternion [x,y,z,w]"""
    parent_index: Final[int]


@dataclass
class BulletRobotJointsInfo:
    """Consolidated static information about all joints for a robot."""

    joint_name_to_info: Mapping[str, BulletJointInfo]
    """Mapping from joint name to BulletJointInfo for each joint of this robot."""
    num_actuated_joints: Final[int]
    """Number of non-fixed (controllable) joints."""
    actuated_joint_names: Final[List[str]]
    """Pybullet joint names of actuated joints only."""
    actuated_joint_ids: Final[List[int]]
    """Pybullet joint ids of actuated joints only."""
    actuated_joint_lower_limits: Final[List[float]]
    actuated_joint_upper_limits: Final[List[float]]
    continuous_joint_names: Final[List[str]]
    """Pybullet joint names of continuous joints only."""
    continuous_joint_ids: Final[List[str]]
    """Pybullet joint ids of continuous joints only."""

    def __init__(self, object_id: int, cid: int = 0):
        """Joint information from pybullet's getJointInfo for all joints.

        Args:
            object_id (int): Pybullet object ID
            cid (int, optional): Pybullet physics client ID. Defaults to 0.
        """

        def _decode_if_required(val):
            if isinstance(val, bytes):
                val = val.decode()
            return val

        joint_name_to_info: Mapping[str, BulletJointInfo] = {}
        num_joints = pb.getNumJoints(object_id, physicsClientId=cid)
        actuated_joint_names = []
        actuated_joint_ids = []
        actuated_joint_lower_limits = []
        actuated_joint_upper_limits = []
        continuous_joint_names = []
        continuous_joint_ids = []
        for i in range(num_joints):
            joint_info_tuple = pb.getJointInfo(object_id, i, physicsClientId=cid)
            joint_id = _decode_if_required(joint_info_tuple[0])
            joint_name = _decode_if_required(joint_info_tuple[1])
            joint_type = _decode_if_required(joint_info_tuple[2])
            q_index = _decode_if_required(joint_info_tuple[3])
            u_index = _decode_if_required(joint_info_tuple[4])
            joint_damping = _decode_if_required(joint_info_tuple[6])
            joint_friction = _decode_if_required(joint_info_tuple[7])
            joint_lower_limit = _decode_if_required(joint_info_tuple[8])
            joint_upper_limit = _decode_if_required(joint_info_tuple[9])
            joint_max_force = _decode_if_required(joint_info_tuple[10])
            joint_max_velocity = _decode_if_required(joint_info_tuple[11])
            link_name = _decode_if_required(joint_info_tuple[12])
            joint_axis = _decode_if_required(joint_info_tuple[13])
            parent_frame_pos = _decode_if_required(joint_info_tuple[14])
            parent_frame_ori = _decode_if_required(joint_info_tuple[15])
            parent_index = _decode_if_required(joint_info_tuple[16])

            joint_name_to_info[joint_name] = BulletJointInfo(
                joint_id=joint_id,
                joint_name=joint_name,
                joint_type=joint_type,
                q_index=q_index,
                u_index=u_index,
                joint_damping=joint_damping,
                joint_friction=joint_friction,
                joint_lower_limit=joint_lower_limit,
                joint_upper_limit=joint_upper_limit,
                joint_max_force=joint_max_force,
                joint_max_velocity=joint_max_velocity,
                link_name=link_name,
                joint_axis=joint_axis,
                parent_frame_pos=parent_frame_pos,
                parent_frame_ori=parent_frame_ori,
                parent_index=parent_index,
            )
            if joint_type == pb.JOINT_FIXED:
                continue

            actuated_joint_ids.append(joint_id)
            actuated_joint_names.append(joint_name)
            actuated_joint_lower_limits.append(joint_lower_limit)
            actuated_joint_upper_limits.append(joint_upper_limit)

            if (
                joint_type == pb.JOINT_REVOLUTE
                and joint_upper_limit == 0
                and joint_lower_limit == 0
            ):
                continuous_joint_names.append(joint_name)
                continuous_joint_ids.append(joint_id)

        self.joint_name_to_info = joint_name_to_info.copy()
        self.num_actuated_joints = len(actuated_joint_names)
        self.actuated_joint_names = actuated_joint_names.copy()
        self.actuated_joint_ids = actuated_joint_ids.copy()
        self.actuated_joint_lower_limits = actuated_joint_lower_limits.copy()
        self.actuated_joint_upper_limits = actuated_joint_upper_limits.copy()
        self.continuous_joint_ids = continuous_joint_ids.copy()
        self.continuous_joint_names = continuous_joint_names.copy()


@dataclass
class BulletLinkInfo:
    """Information about a single link of the robot."""

    mass: Final[float]
    lateral_friction: Final[float]
    local_inertia_diagonal: Final[float]
    local_inertial_pos: Final[float]
    local_inertial_ori: Final[float]
    restitution: Final[float]
    rolling_friction: Final[float]
    spinning_friction: Final[float]
    contact_damping: Final[float]
    contact_stiffness: Final[float]
    body_type: Final[float]
    collision_margin: Final[float]


class BulletRobotLinksInfo:
    """Consolidated information about all links of a robot."""

    link_name_to_info: Mapping[str, BulletLinkInfo]
    link_names: Final[List[str]]
    link_ids: Final[List[int]]

    def __init__(self, object_id: int, cid: int = 0):
        """Link information from links of a pybullet object.

        Args:
            object_id (int): Pybullet object ID
            cid (int, optional): Pybullet physics client ID. Defaults to 0.
        """

        num_joints = pb.getNumJoints(object_id, physicsClientId=cid)
        base_name = pb.getBodyInfo(object_id, physicsClientId=cid)[0].decode()

        link_ids = [-1] + list(range(num_joints))
        link_names = [base_name]
        for i in range(num_joints):
            joint_info_tuple = pb.getJointInfo(object_id, i, physicsClientId=cid)
            link_names.append(joint_info_tuple[12].decode())

        link_name_to_info: Mapping[str, BulletLinkInfo] = {}
        for link_id, link_name in zip(link_ids, link_names):
            link_info_tuple = pb.getDynamicsInfo(
                object_id, link_id, physicsClientId=cid
            )
            link_name_to_info[link_name] = BulletLinkInfo(
                mass=link_info_tuple[0],
                lateral_friction=link_info_tuple[1],
                local_inertia_diagonal=[2],
                local_inertial_pos=link_info_tuple[3],
                local_inertial_ori=link_info_tuple[4],
                restitution=link_info_tuple[5],
                rolling_friction=link_info_tuple[6],
                spinning_friction=link_info_tuple[7],
                contact_damping=link_info_tuple[8],
                contact_stiffness=link_info_tuple[9],
                body_type=link_info_tuple[10],
                collision_margin=link_info_tuple[11],
            )

        self.link_name_to_info = link_name_to_info.copy()
        self.link_names = link_names.copy()
        self.link_ids = link_ids.copy()


class BulletObject(BulletRobotJointsInfo, BulletRobotLinksInfo):
    """Static information about a pybullet object (including joint info and link info)."""

    object_id: int
    name: str
    mass: float
    """Total mass of this object."""
    base_name: str

    def __init__(self, object_id: int, cid: int = 0):
        """Get information about a pybullet object.

        Args:
            object_id (int): Pybullet object ID
            cid (int, optional): Pybullet physics client ID. Defaults to 0.
            verbose (bool, optional): Verbosity flag. Defaults to False.
        """
        self.name = pb.getBodyInfo(object_id, physicsClientId=cid)[1].decode()
        self.base_name = pb.getBodyInfo(object_id, physicsClientId=cid)[0].decode()
        self.object_id = object_id
        BulletRobotJointsInfo.__init__(self, object_id=object_id, cid=cid)
        BulletRobotLinksInfo.__init__(self, object_id=object_id, cid=cid)
        self.mass = np.sum(
            self.link_name_to_info[name].mass for name in self.link_names
        )


class BulletRobot(BulletObject):
    """Robot interface utility for a generic robot in pybullet."""

    name: str
    """Name of the loaded robot as parsed from urdf."""

    def __init__(
        self,
        urdf_path: str,
        cid: int = None,
        run_async: bool = True,
        ee_names: List[str] = None,
        use_fixed_base: bool = True,
        default_joint_positions: List[float] = None,
        place_on_ground: bool = True,
        default_base_position: Vector3D = np.zeros(3),
        default_base_orientation: QuatType = np.array([0, 0, 0, 1]),
        enable_torque_mode: bool = False,
        verbose: bool = True,
        load_ground_plane: bool = True,
        ghost_mode: bool = False,
    ):
        """Robot interface utility for a generic robot in pybullet.

        Args:
            urdf_path (str): Path to urdf file of robot.
            cid (int, optional): Physics client id of pybullet physics engine (if already running).
                If None provided, will create a new physics GUI instance. Defaults to None.
            run_async (bool, optional): Whether to run physics in a separate thread. If set to
                False, step() method has to be called in the main thread. Defaults to True.
            ee_names (List[str], optional): List of end-effectors for the robot. Defaults to None.
            use_fixed_base (bool, optional): Robot will be fixed in the world. Defaults to True.
            default_joint_positions (List[float], optional): Optional starting values for joints.
                Defaults to None. NOTE: These values should be in the order of joints in pybullet.
            place_on_ground (bool): If true, the base position height will automatically be adjusted
                such that the robot is on/above the ground with the given joint positions. Defaults
                to True.
                NOTE: This is a naive implementation. It does not actually check if the collision is
                with the ground object only. It can be with any object in the world.
            default_base_position (Vector3D, optional): Default position of the base of the robot
                in the world frame during start-up. Note that the height value (z) is only used if
                'place_on_ground' is set to False. Defaults to np.zeros(3).
            default_base_orientation (QuatType, optional): Default orientation quaternion in the
                world frame for the base of the robot during start-up. Defaults to
                np.array([0, 0, 0, 1]).
            enable_torque_mode (bool, optional): Flag to enable effort controlling of the robot
                (control commands have to be sent continuously to keep the robot from falling).
                Defaults to True.
            verbose (bool, optional): Verbosity flag for debugging robot info during construction.
                Defaults to True.
            load_ground_plane (bool, optional): If set to True, will load a ground plane in the XY
                plane at origin. Defaults to True.
            ghost_mode (bool, optional): If set to True, the robot will not have any dynamics
                (collision, mass, etc.). The robot will also be partially transparent to denote
                this. Defaults to False.
        """
        self.cid = cid
        if self.cid is None:
            self.cid = pb.connect(pb.GUI_SERVER)

        if load_ground_plane:
            pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb.loadURDF("plane.urdf", physicsClientId=self.cid)

        self.sync_mode = not run_async

        self.default_start_pose = [default_base_position, default_base_orientation]

        self.ee_names = ee_names
        if self.ee_names is None:
            self.ee_names = []

        self.robot_id = pb.loadURDF(
            urdf_path,
            basePosition=self.default_start_pose[0],
            baseOrientation=self.default_start_pose[1],
            useFixedBase=use_fixed_base,
            physicsClientId=self.cid,
        )

        super().__init__(object_id=self.robot_id, cid=self.cid)

        pb.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        fix_pb_link_mass_inertia(body_id=self.robot_id, cid=self.cid)

        self.has_continuous_joints = len(self.continuous_joint_ids) > 0

        self.link_name_to_index = dict(zip(self.link_names, self.link_ids))
        """Mapping from link name to pybullet link id."""
        self.joint_name_to_index = dict(
            zip(
                self.actuated_joint_names,
                self.actuated_joint_ids,
            )
        )
        """Mapping from joint name to pybullet joint id."""

        self.link_id_to_name = dict(zip(self.link_ids, self.link_names))
        """Mapping from pybullet link id to link name."""
        self.joint_id_to_name = dict(
            zip(
                self.actuated_joint_ids,
                self.actuated_joint_names,
            )
        )
        """Mapping from pybullet joint id to joint name."""

        self.ee_ids = [self.link_name_to_index[name] for name in self.ee_names]
        """Pybullet link ids of the end-effector names (in `self.ee_names`)."""

        if verbose:
            self._print_robot_info()

        self.default_joint_positions = np.array(
            default_joint_positions
            if default_joint_positions is not None
            else np.zeros_like(self.actuated_joint_ids)
        )
        self.reset_joints(
            joint_ids=self.actuated_joint_ids,
            joint_positions=self.default_joint_positions,
        )

        self.urdf_path = urdf_path

        self._in_torque_mode = False
        if place_on_ground:
            self._place_robot_on_ground()

        if enable_torque_mode:
            self.set_torque_control_mode()

        self._prev_torque_cmds = dict(
            zip(
                self.actuated_joint_ids,
                np.zeros(self.num_actuated_joints),
            )
        )

        pb.setRealTimeSimulation(0 if self.sync_mode else 1, physicsClientId=self.cid)

        self._ghost_mode = ghost_mode
        if self._ghost_mode:
            for link_id in self.link_ids:
                self.set_robot_transparency(0.3)
                pb.setCollisionFilterGroupMask(
                    self.robot_id, link_id, 0, 0, physicsClientId=self.cid
                )
                pb.changeDynamics(
                    self.robot_id,
                    link_id,
                    mass=0,  # non-dynamic objects have mass 0
                    lateralFriction=0,
                    spinningFriction=0,
                    rollingFriction=0,
                    restitution=0,
                    linearDamping=0,
                    angularDamping=0,
                    contactStiffness=0,
                    contactDamping=0,
                    jointDamping=0,
                    maxJointVelocity=0,
                    physicsClientId=self.cid,
                )

    def refresh_sim_data(self):
        """Refresh all static data of this robot (link info, joint info, etc).
        Needed if dynamics of any link was changed manually. This method is
        called automatically when `change_dynamics()` of this class is called.
        """
        super().__init__(object_id=self.robot_id, cid=self.cid)

    def _print_robot_info(self):
        print("\n")
        print("*" * 100 + "\nSimRobot Info " + "\u2193 " * 20 + "\n" + "*" * 100)
        print("robot ID:              ", self.robot_id)
        print("robot name:            ", self.name)
        print("robot mass:            ", self.mass)
        print("base link name:        ", self.base_name)
        print(
            "link names:            ",
            len(self.link_names),
            self.link_names,
        )
        print(
            "link indexes:          ",
            len(self.link_ids),
            self.link_ids,
        )
        print(
            "link masses:           ",
            len(self.link_ids),
            [self.link_name_to_info[name].mass for name in self.link_names],
        )
        print("end-effectors          ", self.ee_names)
        print(
            "All joints:            ",
            len(self.joint_name_to_info.keys()),
            len(self.joint_name_to_info.keys()),
        )
        print(
            "Actuated joints:       ",
            self.num_actuated_joints,
            self.actuated_joint_names,
        )
        print(
            "joint dampings:        ",
            len(self.actuated_joint_ids),
            [
                self.joint_name_to_info[name].joint_damping
                for name in self.actuated_joint_names
            ],
        )
        print(
            "joint frictions:       ",
            len(self.actuated_joint_ids),
            [
                self.joint_name_to_info[name].joint_friction
                for name in self.actuated_joint_names
            ],
        )
        print(
            "joint lower limits:    ",
            len(self.actuated_joint_lower_limits),
            self.actuated_joint_lower_limits,
        )
        print(
            "joint upper limits:   ",
            len(self.actuated_joint_upper_limits),
            self.actuated_joint_upper_limits,
        )
        print(
            "joint ids:             ",
            len(self.joint_name_to_info.keys()),
            [
                self.joint_name_to_info[name].joint_id
                for name in self.joint_name_to_info.keys()
            ],
        )
        print(
            "actuated joint ids:    ",
            len(self.actuated_joint_ids),
            self.actuated_joint_ids,
        )
        print(
            "continuous joint names:",
            len(self.continuous_joint_names),
            self.continuous_joint_names,
        )
        print(
            "continuous joint ids:  ",
            len(self.continuous_joint_ids),
            self.continuous_joint_ids,
        )
        print("*" * 100 + "\nSimRobot Info " + "\u2191 " * 20 + "\n" + "*" * 100)
        print("\n")

    def step(self):
        """Step simulation (if this object was created with `async=False` argument)."""
        if self.sync_mode:
            pb.stepSimulation(physicsClientId=self.cid)

    def shutdown(self):
        """Cleanly shutdown the pybullet environment."""
        try:
            pb.disconnect(physicsClientId=self.cid)
        except (TypeError, pb.error):
            # raised when pb already dead
            pass

    def toggle_ft_sensor_for_joints(self, joint_ids: List[int], enable: bool = True):
        """Enable/disable a force-torque sensor at the specified joints.

        Args:
            joint_ids (List[int]): List of joint indices.
            enable (bool, optional): True=Enable; False=Disable. Defaults to True.
        """
        if isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        for jid in joint_ids:
            pb.enableJointForceTorqueSensor(
                self.robot_id, jid, enable, physicsClientId=self.cid
            )

    def set_robot_transparency(self, alpha: float):
        """Set the visual transparency of the robot.

        Args:
            alpha (float): Float value between 0 (invisible) to 1 (opaque).
        """
        vis_data = pb.getVisualShapeData(self.robot_id, physicsClientId=self.cid)
        for data in vis_data:
            pb.changeVisualShape(
                self.robot_id,
                data[1],
                rgbaColor=(data[7][0], data[7][1], data[7][2], alpha),
                physicsClientId=self.cid,
            )

    def _place_robot_on_ground(
        self,
        move_resolution: float = 0.01,
        default_position: Vector3D = None,
        default_orientation: QuatType = None,
    ):
        pb.setRealTimeSimulation(0)
        in_collision = True
        if default_position is None:
            default_position = [
                self.default_start_pose[0][0],
                self.default_start_pose[0][1],
                -move_resolution,
            ]
        if default_orientation is None:
            default_orientation = self.default_start_pose[1]

        joint_pos = self.get_joint_states(joint_ids=self.actuated_joint_ids)[0].copy()
        while in_collision:
            default_position[2] += move_resolution
            self.reset_base_pose(
                position=default_position, orientation=default_orientation
            )
            self.reset_joints(
                joint_ids=self.actuated_joint_ids,
                joint_positions=joint_pos,
            )
            pb.stepSimulation(physicsClientId=self.cid)
            in_collision = np.any(self.get_contact_states_of_links(self.link_ids))
        self.default_start_pose[0] = np.array(default_position)

    def set_position_control_mode(self):
        """Set control mode to position-based (robot will stay in place if no commands sent)."""
        pb.setJointMotorControlArray(
            self.robot_id,
            self.actuated_joint_ids,
            pb.POSITION_CONTROL,
            targetPositions=self.get_actuated_joint_positions(),
            physicsClientId=self.cid,
        )
        self._in_torque_mode = False
        print("SimRobot position control mode enabled!")

    def set_torque_control_mode(self):
        """Set control mode to effort-based (robot will NOT stay in place if no commands sent)."""
        pb.setJointMotorControlArray(
            self.robot_id,
            self.actuated_joint_ids,
            pb.VELOCITY_CONTROL,
            forces=[0.0] * self.num_actuated_joints,
            physicsClientId=self.cid,
        )
        self._in_torque_mode = True
        print("SimRobot effort control mode enabled!")

    def reset_base_pose(
        self,
        position: Vector3D = None,
        orientation: QuatType = None,
        base_linear_velocity: Vector3D = None,
        base_angular_velocity: Vector3D = None,
    ):
        """Reset the robot's base pose to default pose.

        Args:
            position (Vector3D, optional): If specified, sets robot to this position. Defaults to
                None.
            orientation (QuatType, optional): If specified, sets robot to this orientation.
                Quaternion order: x,y,z,w. Defaults to None.
        """
        if position is None:
            position = self.default_start_pose[0]
        if orientation is None:
            orientation = self.default_start_pose[1]
        if base_linear_velocity is None:
            base_linear_velocity = np.zeros(3)
        if base_angular_velocity is None:
            base_angular_velocity = np.zeros(3)
        pb.resetBasePositionAndOrientation(
            self.robot_id, position, orientation, physicsClientId=self.cid
        )

    def reset_joints(
        self,
        joint_ids: List[int],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray = None,
    ):
        """Reset joints of the robot (no controller).

        Args:
            joint_ids (List[int]): List of joints to reset.
            joint_positions (np.ndarray): Positions to set the specified joints.
            joint_velocities (np.ndarray, optional): Joint velocities. Defaults to None.
        """
        if joint_velocities is None:
            joint_velocities = np.zeros(len(joint_ids))
        for n, jid in enumerate(joint_ids):
            pb.resetJointState(
                self.robot_id,
                jid,
                joint_positions[n],
                joint_velocities[n],
                physicsClientId=self.cid,
            )

    def reset_actuated_joint_positions(
        self,
        joint_positions: np.ndarray = None,
        joint_names: List[str] = None,
        joint_velocities: np.ndarray = None,
    ):
        """Reset the joint positions of the robot.

        Args:
            joint_positions (np.ndarray, optional): If specified, sets joints to these values,
                otherwise uses default. Defaults to None.
            joint_names (List[str], optional): Joint names in the order the position values were
                given. If None provided, uses default order of `joints_info.actuated_joint_names`.
        """
        if joint_positions is None:
            joint_positions = self.default_joint_positions
        if joint_names is None:
            joint_names = self.actuated_joint_names
        joint_ids = self.get_joint_ids(joint_names=joint_names)
        self.reset_joints(
            joint_ids=joint_ids,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
        )

    def get_joint_name(self, joint_id: int) -> str:
        """Get name of joint given it's pybullet joint index.

        Args:
            joint_id (int): Pybullet joint id.

        Returns:
            str: Corresponding joint name.
        """
        return self.joint_id_to_name[joint_id]

    def get_joint_id(self, joint_name: str) -> int:
        """Get the joint id of the specied joint.

        Args:
            joint_name (str): Name of joint.

        Returns:
            int: Pybullet joint index.
        """
        return self.joint_name_to_index[joint_name]

    def get_joint_ids(self, joint_names: List[str]) -> List[int]:
        """Get the joint ids of the specied joints.

        Args:
            joint_names (List[str]): Joint names.

        Returns:
            List[int]: List of Pybullet joint ids.
        """
        return [self.joint_name_to_index[joint_name] for joint_name in joint_names]

    def get_link_name(self, link_id: int) -> str:
        """Get name of link given it's pybullet link index.

        Args:
            link_id (int): Pybullet link id.

        Returns:
            str: Corresponding link name.
        """
        return self.link_id_to_name[link_id]

    def get_link_id(self, link_name: str) -> int:
        """Get the link id of the specied link.

        Args:
            link_name (str): Name of link.

        Returns:
            int: Pybullet link index.
        """
        return self.link_name_to_index[link_name]

    def get_base_com_position(self) -> Vector3D:
        world_trans_com = self.get_base_pose()

        com_trans_local = pb.invertTransform(
            position=self.link_name_to_info[self.base_name].local_inertial_pos,
            orientation=self.link_name_to_info[self.base_name].local_inertial_ori,
            physicsClientId=self.cid,
        )
        world_trans_local = pb.multiplyTransforms(
            positionA=world_trans_com[0],
            orientationA=world_trans_com[1],
            positionB=com_trans_local[0],
            orientationB=com_trans_local[1],
            physicsClientId=self.cid,
        )
        return np.array(world_trans_local[0])

    def get_base_pose(self) -> Tuple[Vector3D, QuatType]:
        p, q = pb.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.cid)
        if np.isnan(q[0]):
            # BUG: occasional nans in quaternion retrieved from pybullet
            # https://github.com/bulletphysics/bullet3/issues/976
            raise RuntimeError("Pybullet pose retrieval returned a nan quaternion")
        return np.array(p), np.array(q)

    def get_base_velocity(self) -> Tuple[Vector3D, Vector3D]:
        lin, ang = pb.getBaseVelocity(self.robot_id, physicsClientId=self.cid)
        return np.array(lin), np.array(ang)

    def change_dynamics(self, link_name: str, **kwargs):
        """Exposes the `pb.changeDynamics` pybullet API for this robot. Also
        updates the robot, link and joint info (accessed via `object_info` dict,
        and other exposed properties related to joints and links for this robot)
        after making the change.

        Will update `self` attributes automatically.

        NOTE: Not to be used in high-speed loops as this operation can be heavy.
        """
        kwargs.pop("bodyUniqueId", None)
        kwargs.pop("linkIndex", None)
        kwargs.pop("physicsClientId", None)
        pb.changeDynamics(
            bodyUniqueId=self.robot_id,
            linkIndex=self.get_link_id(link_name=link_name),
            physicsClientId=self.cid,
            **kwargs,
        )
        self.refresh_sim_data()

    def get_actuated_joint_positions(
        self, actuated_joint_names: List[str] = None
    ) -> np.ndarray:
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(
            joint_ids=self.get_joint_ids(actuated_joint_names)
        )[0]

    def get_actuated_joint_velocities(
        self, actuated_joint_names: List[str] = None
    ) -> np.ndarray:
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(
            joint_ids=self.get_joint_ids(actuated_joint_names)
        )[1]

    def get_actuated_joint_torques(
        self, actuated_joint_names: List[str] = None
    ) -> np.ndarray:
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        return self.get_joint_states(
            joint_ids=self.get_joint_ids(actuated_joint_names)
        )[2]

    def get_jacobian(self, ee_link_name: str, joint_angles=None) -> np.ndarray:
        if joint_angles is None:
            joint_angles = self.get_actuated_joint_positions()

        linear_jacobian, angular_jacobian = pb.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.get_link_id(ee_link_name),
            localPosition=[0.0, 0.0, 0.0],
            objPositions=joint_angles.tolist(),
            objVelocities=[0] * len(joint_angles),
            objAccelerations=[0] * len(joint_angles),
            physicsClientId=self.cid,
        )

        return np.vstack([np.array(linear_jacobian), np.array(angular_jacobian)])

    def get_link_pose(self, link_id: int) -> Tuple[Vector3D, QuatType]:
        if link_id == -1:
            return self.get_base_pose()
        p, o = pb.getLinkState(self.robot_id, link_id, physicsClientId=self.cid)[4:6]
        return np.array(p), np.array(o)

    def get_link_velocity(self, link_id: int) -> Tuple[Vector3D, Vector3D]:
        link_state = pb.getLinkState(
            self.robot_id, link_id, computeLinkVelocity=1, physicsClientId=self.cid
        )

        return np.asarray(link_state[6]), np.asarray(link_state[7])

    def get_link_com_pose(self, link_id: int) -> Tuple[Vector3D, QuatType]:
        if link_id == -1:
            # NOTE: probably not valid for base
            return self.get_base_com_position(), self.get_base_pose()[1]
        p, o = pb.getLinkState(self.robot_id, link_id, physicsClientId=self.cid)[:2]
        return np.array(p), np.array(o)

    def get_ee_contact_states(self, ee_names: List[str] = None) -> np.ndarray:
        if ee_names is None:
            ee_names = self.ee_names
        return self.get_contact_states_of_links(
            link_ids=[self.get_link_id(link_name=name) for name in ee_names]
        )

    def get_contact_states_of_links(self, link_ids: List[int]) -> np.ndarray:
        """Return a list denoting whether the specified links are in contact
        or not. (1 for contact; 0 for not in contact)

        Args:
            link_ids (List[int]): List of link indices.

        Returns:
            np.ndarray: List of binary contact state values.
        """
        return np.array(
            [
                int(
                    len(
                        pb.getContactPoints(
                            bodyA=self.robot_id,
                            linkIndexA=idx,
                            physicsClientId=self.cid,
                        )
                    )
                    > 0
                )
                for idx in link_ids
            ]
        )

    def toggle_self_collision(self, enable: bool):
        """Enable/disable self-collision for this robot.

        Args:
            enable (bool): True or false.
        """
        for n, l_id in enumerate(self.link_ids):
            if n < len(self.link_ids) - 1:
                for l2_id in self.link_ids[n:]:
                    pb.setCollisionFilterPair(
                        self.robot_id,
                        self.robot_id,
                        l_id,
                        l2_id,
                        int(enable),
                        physicsClientId=self.cid,
                    )

    def get_physics_parameters(self) -> Dict[str, Any]:
        """Get all values from pybullet's getPhysicsEngineParameters API."""
        return pb.getPhysicsEngineParameters(physicsClientId=self.cid)

    def get_robot_states(
        self, actuated_joint_names: List[str] = None, ee_names: List[str] = None
    ) -> Dict[str, Any]:
        """Get a dictionary of current robot state in the simulation.

        Args:
            actuated_joint_names (List[str], optional): List of joint names that are
                to be included in the robot state info. Defaults to None (uses all
                actuated joints).
            ee_names (List[str], optional): List of end-effector names that are
                to be included in the robot state info. Defaults to None (uses all
                end-effectors specified during construction).

        Returns:
            Dict[str, Any]: Dictionary of key to value. (see below)
                "base_position" -> Vector3D
                "base_com_position" -> Vector3D
                "base_quaternion" -> QuatType
                "base_velocity_linear" -> Vector3D
                "base_velocity_angular" -> Vector3D
                "actuated_joint_positions" -> np.ndarray
                "actuated_joint_velocities" -> np.ndarray
                "actuated_joint_torques" -> np.ndarray
                "joint_order" -> List[str]
                "ee_order" -> List[str]
                "ee_contact_states" -> List[int]

        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
            joint_ids = self.actuated_joint_ids
        else:
            joint_ids = self.get_joint_ids(joint_names=actuated_joint_names)
        if ee_names is None:
            ee_ids = self.ee_ids
            ee_names = self.ee_names
        else:
            ee_ids = [self.get_link_id(link_name=name) for name in ee_names]

        base_pose = self.get_base_pose()
        base_vel = self.get_base_velocity()
        joint_states = self.get_joint_states(joint_ids=joint_ids)
        return {
            "base_position": base_pose[0],
            "base_com_position": self.get_base_com_position(),
            "base_quaternion": base_pose[1],
            "base_velocity_linear": base_vel[0],
            "base_velocity_angular": base_vel[1],
            "actuated_joint_positions": joint_states[0],
            "actuated_joint_velocities": joint_states[1],
            "actuated_joint_torques": joint_states[2],
            "joint_order": actuated_joint_names,
            "ee_order": ee_names,
            "ee_contact_states": self.get_contact_states_of_links(ee_ids),
        }

    def get_full_joint_states(
        self, joint_ids: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get joint positions, velocities, reaction wrenches, and efforts.

        Args:
            joint_ids (List[int], Optional): List of joint ids (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: joint positions, joint velocities, measured
                joint force-torque values (if FT sensor enabled for the joint), joint efforts.
        """
        if joint_ids is None:
            joint_ids = self.actuated_joint_ids
        dof = len(joint_ids)
        q, v, ft, tau = np.zeros(dof), np.zeros(dof), np.zeros([dof, 6]), np.zeros(dof)
        for n, idx in enumerate(joint_ids):
            q[n], v[n], ft[n, :], tau[n] = pb.getJointState(
                self.robot_id, idx, physicsClientId=self.cid
            )
            if idx in self.continuous_joint_ids:
                q[n] = wrap_angle(q[n])
            if self._in_torque_mode:
                tau[n] = self._prev_torque_cmds[idx]

        return q, v, ft, tau

    def get_joint_states(
        self, joint_ids: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get joint positions, velocities and efforts.

        Args:
            joint_ids (List[int], Optional): List of joint ids (optional)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: joint positions, joint velocities, joint
                efforts
        """
        q, v, _, tau = self.get_full_joint_states(joint_ids=joint_ids)
        return q, v, tau

    def get_joint_ft_measurements(self, joint_ids: List[int] = None) -> np.ndarray:
        """Get measured joint force-torque values (if FT sensor enabled for the joint).

        Args:
            joint_ids (List[int], Optional): List of joint ids (optional)

        Returns:
            np.ndarray: measured joint force-torque values (if FT sensor enabled for the joint).
        """
        return self.get_full_joint_states(joint_ids=joint_ids)[2]

    def set_joint_velocities(
        self, cmd: np.ndarray, actuated_joint_names: List[str] = None
    ):
        # NOTE: Not tested
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        pb.setJointMotorControlArray(
            self.robot_id,
            jids,
            controlMode=pb.VELOCITY_CONTROL,
            targetVelocities=cmd,
            physicsClientId=self.cid,
        )

    def set_joint_torques(
        self, cmd: np.ndarray, actuated_joint_names: List[str] = None
    ):
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        pb.setJointMotorControlArray(
            self.robot_id,
            jids,
            pb.TORQUE_CONTROL,
            forces=cmd,
            physicsClientId=self.cid,
        )

    def set_joint_positions_delta(
        self, cmd: np.ndarray, actuated_joint_names: List[str] = None
    ):
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        pb.setJointMotorControlArray(
            self.robot_id,
            jids,
            controlMode=pb.POSITION_CONTROL,
            targetVelocities=cmd,
            physicsClientId=self.cid,
        )

    def set_joint_positions(
        self,
        cmd: np.ndarray,
        actuated_joint_names: List[str] = None,
        vels: np.ndarray = None,
    ):
        kwargs = {}
        if vels is not None:
            kwargs["targetVelocities"] = vels
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        pb.setJointMotorControlArray(
            self.robot_id,
            jids,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=cmd,
            physicsClientId=self.cid,
            **kwargs,
        )

    def compute_joint_pd_error(
        self,
        joint_ids: List[int],
        q_des: np.ndarray,
        dq_des: np.ndarray,
        Kp: float | np.ndarray,
        Kd: float | np.ndarray,
    ):
        curr_q, curr_dq, _ = self.get_joint_states(joint_ids=joint_ids)

        p_term = np.zeros_like(curr_q)
        if q_des is not None:
            p_term = Kp * (q_des - curr_q)
            if self.has_continuous_joints:
                for n, jid in enumerate(joint_ids):
                    if jid in self.continuous_joint_ids:
                        p_term[n] = Kp[n] * wrap_angle(q_des[n] - curr_q[n])

        d_term = np.zeros_like(curr_dq)
        if dq_des is not None:
            d_term = Kd * (dq_des - curr_dq)

        return p_term + d_term

    def set_actuated_joint_commands(
        self,
        actuated_joint_names: List[str] = None,
        q: float | np.ndarray = 0,
        dq: float | np.ndarray = 0,
        Kp: float | np.ndarray = 0,
        Kd: float | np.ndarray = 0,
        tau: float | np.ndarray = 0,
    ):
        """Set PVT-PD command for the robot.
        (Position-velocity-torque + PD gains)

        NOTE: only valid if the robot was instantiated in torque mode (use
        `enable_torque_mode=True` during construction.

        Args:
            actuated_joint_names (List[str], optional): List of joint names to control.
                Defaults to None (uses all actuated joints).
            q (float | np.ndarray, optional): Position commands. Defaults to 0.
            dq (float | np.ndarray, optional): Velocity commands. Defaults to 0.
            Kp (float | np.ndarray, optional): Stiffness gains per joint. Defaults to 0.
            Kd (float | np.ndarray, optional): Damping gains per joint. Defaults to 0.
            tau (float | np.ndarray, optional): Feedforward torque commands. Defaults to 0.
        """
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        if self._in_torque_mode:
            tau_cmd = (
                self.compute_joint_pd_error(
                    joint_ids=jids, q_des=q, dq_des=dq, Kp=Kp, Kd=Kd
                )
                + tau
            )
            pb.setJointMotorControlArray(
                self.robot_id,
                jids,
                pb.TORQUE_CONTROL,
                forces=tau_cmd,
                physicsClientId=self.cid,
            )
            for n, jid in enumerate(jids):
                self._prev_torque_cmds[jid] = tau_cmd[n]
        else:
            # tau = [self._max_actuator_force] * len(actuated_joint_names)
            if isinstance(dq, Number):
                dq = [dq] * len(actuated_joint_names)
            pb.setJointMotorControlArray(
                self.robot_id,
                jids,
                pb.POSITION_CONTROL,
                targetPositions=q,
                targetVelocities=dq,
                # forces=tau,
                # positionGains=Kp,
                # velocityGains=Kd,
                physicsClientId=self.cid,
            )
