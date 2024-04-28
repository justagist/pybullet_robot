"""Pybullet robot interface"""

from typing import List, Dict, Any, Tuple, TypeAlias
from threading import Lock
from numbers import Number
from pprint import pprint
import pybullet as pb
import numpy as np

# pylint: disable=I1101, R0914, C0302, R0902, R0904, R0913

QuatType: TypeAlias = np.ndarray
"""Numpy array representating quaternion in format [x,y,z,w]"""
Vector3D: TypeAlias = np.ndarray  # pylint: disable=C0103
"""Numpy array representating 3D cartesian vector in format [x,y,z]"""


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap the provided angle(s) (radians) to be within -pi and pi.

    Args:
        angle (float | np.ndarray): Input angle (or array of angles) in radians.

    Returns:
        float | np.ndarray: Output after wrapping.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_pybullet_object_joint_info(
    object_id: int, cid: int = 0, verbose: bool = False
) -> Dict[str, Any]:
    """Get dictionarised version of joint information from pybullet's getJointInfo for all joints.

    Args:
        object_id (int): Pybullet object ID
        cid (int, optional): Pybullet physics client ID. Defaults to 0.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    Returns:
        Dict[str, Any]: Key-value pairs of joint info.
            Keys:
                "joint_index",
                "joint_name",
                "joint_type",
                "q_index",
                "u_index",
                "flags",
                "joint_damping",
                "joint_friction",
                "joint_lower_limit",
                "joint_upper_limit",
                "joint_max_force",
                "joint_max_velocity",
                "link_name",
                "joint_axis",
                "parent_frame_pos",
                "parent_frame_ori",
                "parent_index",
                "num_actuated_joint",
                "actuated_joint_name",
                "actuated_joint_index",
                "actuated_joint_lower_limit",
                "actuated_joint_upper_limit"

    """
    PYBULLET_JOINT_INFO_KEYS = [
        "joint_index",
        "joint_name",
        "joint_type",
        "q_index",
        "u_index",
        "flags",
        "joint_damping",
        "joint_friction",
        "joint_lower_limit",
        "joint_upper_limit",
        "joint_max_force",
        "joint_max_velocity",
        "link_name",
        "joint_axis",
        "parent_frame_pos",
        "parent_frame_ori",
        "parent_index",
    ]

    num_joints = pb.getNumJoints(object_id, physicsClientId=cid)
    joint_info_values = [[] for _ in range(len(PYBULLET_JOINT_INFO_KEYS))]
    for i in range(num_joints):
        joint_info_tuple = pb.getJointInfo(object_id, i, physicsClientId=cid)
        for joint_info_value, joint_info_item in zip(
            joint_info_values, joint_info_tuple
        ):
            if isinstance(joint_info_item, bytes):
                joint_info_item = joint_info_item.decode()
            joint_info_value.append(joint_info_item)
    joint_info = dict(zip(PYBULLET_JOINT_INFO_KEYS, joint_info_values))
    joint_info["num_joint"] = num_joints
    actuated_joint_name = []
    actuated_joint_index = []
    actuated_joint_lower_limits = []
    actuated_joint_upper_limits = []
    continuous_joint_name = []
    continuous_joint_index = []

    for n, name in enumerate(joint_info["joint_name"]):
        jtype = joint_info["joint_type"][n]
        if jtype == pb.JOINT_FIXED:
            continue

        j_upper_lim = joint_info["joint_upper_limit"][n]
        j_lower_lim = joint_info["joint_lower_limit"][n]
        actuated_joint_index.append(n)
        actuated_joint_name.append(name)
        actuated_joint_lower_limits.append(j_lower_lim)
        actuated_joint_upper_limits.append(j_upper_lim)

        if jtype == pb.JOINT_REVOLUTE and j_upper_lim == 0 and j_lower_lim == 0:
            continuous_joint_name.append(name)
            continuous_joint_index.append(n)

    joint_info["num_actuated_joint"] = len(actuated_joint_name)
    joint_info["actuated_joint_name"] = actuated_joint_name
    joint_info["actuated_joint_index"] = actuated_joint_index
    joint_info["continuousJointName"] = continuous_joint_name
    joint_info["continuousJointIndex"] = continuous_joint_index
    joint_info["actuated_joint_lower_limit"] = actuated_joint_lower_limits
    joint_info["actuated_joint_upper_limit"] = actuated_joint_upper_limits

    if verbose:
        print("joint_info:")
        pprint(joint_info)
    return joint_info


def get_pybullet_object_link_info(
    object_id: int, cid: int = 0, verbose: bool = False
) -> Dict[str, Any]:
    """Get dictionarised version of link information from links of a pybullet object.

    Args:
        object_id (int): Pybullet object ID
        cid (int, optional): Pybullet physics client ID. Defaults to 0.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    Returns:
        Dict[str, Any]: Key-value pairs of joint info.
            Keys:
                "mass",
                "lateral_friction",
                "local_inertia_diagonal",
                "local_inertial_pos",
                "local_inertial_ori",
                "restitution",
                "rolling_friction",
                "spinning_friction",
                "contact_damping",
                "contact_stiffness",
                "body_type",
                "collision_margin",
                "index",
                "name"

    """
    PYBULLET_LINK_INFO_KEYS = [
        "mass",
        "lateral_friction",
        "local_inertia_diagonal",
        "local_inertial_pos",
        "local_inertial_ori",
        "restitution",
        "rolling_friction",
        "spinning_friction",
        "contact_damping",
        "contact_stiffness",
        "body_type",
        "collision_margin",
    ]

    num_joints = pb.getNumJoints(object_id, physicsClientId=cid)
    base_name = pb.getBodyInfo(object_id, physicsClientId=cid)[0].decode()

    link_ids = [-1] + list(range(num_joints))
    link_names = [base_name]
    for i in range(num_joints):
        joint_info_tuple = pb.getJointInfo(object_id, i, physicsClientId=cid)
        link_names.append(joint_info_tuple[12].decode())

    link_info_values = [[] for _ in range(len(PYBULLET_LINK_INFO_KEYS))]
    for link_id in link_ids:
        link_info_tuple = pb.getDynamicsInfo(object_id, link_id, physicsClientId=cid)
        for link_info_value, link_info_item in zip(link_info_values, link_info_tuple):
            link_info_value.append(link_info_item)
    link_info = dict(zip(PYBULLET_LINK_INFO_KEYS, link_info_values))

    link_info["index"] = link_ids
    link_info["name"] = link_names

    if verbose:
        print("link_info:")
        pprint(link_info)

    return link_info


def get_pybullet_object_info(
    object_id: int, cid: int = 0, verbose: bool = True
) -> Dict[str, Any]:
    """Get dictionarised version of information about a pybullet object.

    Args:
        object_id (int): Pybullet object ID
        cid (int, optional): Pybullet physics client ID. Defaults to 0.
        verbose (bool, optional): Verbosity flag. Defaults to False.

    Returns:
        Dict[str, Any]: Key-value pairs of joint info.
            Keys:
                "object_id",
                "object_name",
                "object_mass",
                "base_name",
                "joint_info",
                "link_info",

    """
    object_name = pb.getBodyInfo(object_id, physicsClientId=cid)[1].decode()
    base_name = pb.getBodyInfo(object_id, physicsClientId=cid)[0].decode()
    joint_info = get_pybullet_object_joint_info(object_id, cid=cid)
    link_info = get_pybullet_object_link_info(object_id, cid=cid)
    object_mass = np.sum(link_info["mass"])
    object_info = {
        "object_id": object_id,
        "object_name": object_name,
        "object_mass": object_mass,
        "base_name": base_name,
        "joint_info": joint_info,
        "link_info": link_info,
    }

    if verbose:
        print("\n" + "*" * 100 + "\nPybullet Info " + "\u2193 " * 20 + "\n" + "*" * 100)
        print("object_info:")
        pprint(object_info)
        print("*" * 100 + "\nPybullet Info " + "\u2191 " * 20 + "\n" + "*" * 100)
    return object_info


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
    for link_index in link_ids:
        link_info = pb.getDynamicsInfo(body_id, link_index, physicsClientId=cid)
        if link_info[0] == 1.0 and link_info[2] == (1.0, 1.0, 1.0):
            pb.changeDynamics(
                bodyUniqueId=body_id,
                linkIndex=link_index,
                mass=mass,
                localInertiaDiagonal=local_inertia_diagonal,
                physicsClientId=cid,
            )


class BulletRobot:
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
            import pybullet_data

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
        pb.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        self._state_mutex = Lock()

        fix_pb_link_mass_inertia(body_id=self.robot_id, cid=self.cid)

        # retrieve all the info about the robot and populate attributes
        self._retrieve_robot_info()

        if verbose:
            self._print_robot_info()

        self.default_joint_positions = (
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
            zip(self.actuated_joint_ids, np.zeros(self.num_of_actuated_joints))
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

    def _retrieve_robot_info(self):
        self._object_info = get_pybullet_object_info(
            self.robot_id, cid=self.cid, verbose=False
        )
        self.name = self._object_info["object_name"]
        self.mass = self._object_info["object_mass"]
        self.base_name = self._object_info["base_name"]
        self.link_names = self._object_info["link_info"]["name"]
        self.link_ids = self._object_info["link_info"]["index"]
        self.link_masses = self._object_info["link_info"]["mass"]
        self.num_joints = self._object_info["joint_info"]["num_joint"]
        self.joint_names = self._object_info["joint_info"]["joint_name"]
        self.joint_ids = self._object_info["joint_info"]["joint_index"]
        self.joint_dampings = self._object_info["joint_info"]["joint_damping"]
        self.joint_frictions = self._object_info["joint_info"]["joint_friction"]
        self.joint_lower_limits = self._object_info["joint_info"]["joint_lower_limit"]
        self.joint_upper_limits = self._object_info["joint_info"]["joint_upper_limit"]
        self.joint_neutral_positions = (
            np.array(self.joint_lower_limits) + np.array(self.joint_upper_limits)
        ) / 2
        self.num_of_actuated_joints = self._object_info["joint_info"][
            "num_actuated_joint"
        ]
        self.actuated_joint_names = self._object_info["joint_info"][
            "actuated_joint_name"
        ]
        self.actuated_joint_ids = self._object_info["joint_info"][
            "actuated_joint_index"
        ]
        self.actuated_joint_lower_limits = self._object_info["joint_info"][
            "actuated_joint_lower_limit"
        ]
        self.actuated_joint_upper_limits = self._object_info["joint_info"][
            "actuated_joint_upper_limit"
        ]
        self.continuous_joint_ids = self._object_info["joint_info"][
            "continuousJointIndex"
        ]
        self.continuous_joint_names = self._object_info["joint_info"][
            "continuousJointName"
        ]

        self.has_continuous_joints = len(self.continuous_joint_ids) > 0

        self.actuated_joint_neutral_positions = (
            np.array(self.actuated_joint_lower_limits)
            + np.array(self.actuated_joint_upper_limits)
        ) / 2

        self.link_name_to_index = dict(zip(self.link_names, self.link_ids))
        self.joint_name_to_index = dict(zip(self.joint_names, self.joint_ids))
        self.ee_ids = [self.link_name_to_index[name] for name in self.ee_names]

    @property
    def object_info(self) -> Dict[str, Any]:
        """Dictionary with useful static information about the loaded robot."""
        return self._object_info

    def refresh_sim_data(self):
        self._retrieve_robot_info()

    def _print_robot_info(self):
        print("\n")
        print("*" * 100 + "\nSimRobot Info " + "\u2193 " * 20 + "\n" + "*" * 100)
        print("robot ID:              ", self.robot_id)
        print("robot name:            ", self.name)
        print("robot mass:            ", self.mass)
        print("base link name:        ", self.base_name)
        print("link names:            ", len(self.link_names), self.link_names)
        print("link indexes:          ", len(self.link_ids), self.link_ids)
        print("link masses:           ", len(self.link_masses), self.link_masses)
        print("end-effectors          ", self.ee_names)

        print("num of joints:         ", self.num_joints)
        print("num of actuated joints:", self.num_of_actuated_joints)

        print("joint names:           ", len(self.joint_names), self.joint_names)
        print("joint indexes:         ", len(self.joint_ids), self.joint_ids)
        print("joint dampings:        ", len(self.joint_dampings), self.joint_dampings)
        print(
            "joint frictions:       ", len(self.joint_frictions), self.joint_frictions
        )
        print(
            "joint lower limits:    ",
            len(self.joint_lower_limits),
            self.joint_lower_limits,
        )
        print(
            "joint higher limits:   ",
            len(self.joint_upper_limits),
            self.joint_upper_limits,
        )

        print(
            "actuated joint names:  ",
            len(self.actuated_joint_names),
            self.actuated_joint_names,
        )
        print(
            "actuated joint indexes:",
            len(self.actuated_joint_ids),
            self.actuated_joint_ids,
        )
        print(
            "continuous joint names:  ",
            len(self.continuous_joint_names),
            self.continuous_joint_names,
        )
        print(
            "continuous joint indexes:",
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
        try:
            pb.disconnect(physicsClientId=self.cid)
        except (TypeError, pb.error):
            # raised when pb already dead
            pass

    def toggle_ft_sensor_for_joints(self, joint_ids: List[int], enable: bool = True):
        if isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        for jid in joint_ids:
            pb.enableJointForceTorqueSensor(
                self.robot_id, jid, enable, physicsClientId=self.cid
            )

    def set_robot_transparency(self, alpha: float):
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
        # print(f"{self.__class__.__name__}: Trying to place robot on the ground.")
        while in_collision:
            default_position[2] += move_resolution
            self.reset_base_pose(
                position=default_position, orientation=default_orientation
            )
            self.reset_joints(
                joint_ids=self.actuated_joint_ids, joint_positions=joint_pos
            )
            pb.stepSimulation(physicsClientId=self.cid)
            in_collision = np.any(self.get_contact_states_of_links(self.link_ids))
        # print(f"{self.__class__.__name__}: Robot should be just above the ground now.")
        self.default_start_pose[0] = np.array(default_position)

    def set_position_control_mode(self):
        """Set control mode to position-based (robot will stay in place if no commands sent)."""
        pb.setJointMotorControlArray(
            self.robot_id,
            self.joint_ids,
            pb.POSITION_CONTROL,
            targetPositions=self.joint_neutral_positions,
            physicsClientId=self.cid,
        )
        self._in_torque_mode = False
        print("SimRobot position control mode enabled!")

    def get_physics_parameters(self) -> Dict[str, Any]:
        return pb.getPhysicsEngineParameters(physicsClientId=self.cid)

    def set_torque_control_mode(self):
        """Set control mode to effort-based (robot will NOT stay in place if no commands sent)."""
        pb.setJointMotorControlArray(
            self.robot_id,
            self.actuated_joint_ids,
            pb.VELOCITY_CONTROL,
            forces=[0.0] * self.num_of_actuated_joints,
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
            position (Vector3D, optional): If specified, sets robot to this position. Defaults to None.
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
                given. If None provided, uses default order of self.actuated_joint_names.
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

    def get_object_id(self) -> int:
        return self.robot_id

    def get_base_link_name(self) -> str:
        return pb.getBodyInfo(self.robot_id, physicsClientId=self.cid)[0].decode()

    def get_joint_name(self, joint_id: int) -> str:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[
            1
        ].decode()

    def get_joint_id(self, joint_name: str) -> int:
        return self.joint_name_to_index[joint_name]

    def get_joint_ids(self, joint_names: List[str]) -> List[int]:
        return [self.joint_name_to_index[joint_name] for joint_name in joint_names]

    def get_link_name(self, link_id: int) -> str:
        if link_id == -1:
            return self.get_base_link_name()
        else:
            return pb.getJointInfo(self.robot_id, link_id, physicsClientId=self.cid)[
                12
            ].decode()

    def get_link_index(self, link_name: str) -> int:
        return self.link_name_to_index[link_name]

    def get_joint_type(self, joint_id: int) -> str:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[2]

    def get_joint_damping(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[6]

    def get_joint_friction(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[7]

    def get_joint_lower_limit_position(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[8]

    def get_joint_upper_limit_position(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[9]

    def get_joint_max_force(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[10]

    def get_joint_max_velocity(self, joint_id: int) -> float:
        return pb.getJointInfo(self.robot_id, joint_id, physicsClientId=self.cid)[11]

    def get_link_mass(self, link_id: int) -> float:
        return pb.getDynamicsInfo(self.robot_id, link_id, physicsClientId=self.cid)[0]

    def get_link_local_inertial_transform(
        self, link_id: int
    ) -> Tuple[np.ndarray, QuatType]:
        return pb.getDynamicsInfo(self.robot_id, link_id, physicsClientId=self.cid)[3:5]

    def get_link_local_inertial_position(self, link_id: int) -> np.ndarray:
        return np.array(
            pb.getDynamicsInfo(self.robot_id, link_id, physicsClientId=self.cid)[3]
        )

    def get_link_local_inertial_quaternion(self, link_id: int) -> QuatType:
        return np.array(
            pb.getDynamicsInfo(self.robot_id, link_id, physicsClientId=self.cid)[4]
        )

    # Get Base states
    def get_base_local_inertia_transform(self) -> Tuple[Vector3D, QuatType]:
        return self.get_link_local_inertial_transform(-1)

    def get_base_local_inertia_position(self) -> Vector3D:
        return self.get_link_local_inertial_position(-1)

    def get_base_local_inertia_quaternion(self) -> QuatType:
        return self.get_link_local_inertial_quaternion(-1)

    def get_base_com_position(self) -> Vector3D:
        worldTransCom = self.get_base_pose()

        localTransCom = self.get_base_local_inertia_transform()
        comTransLocal = pb.invertTransform(
            position=localTransCom[0],
            orientation=localTransCom[1],
            physicsClientId=self.cid,
        )
        worldTransLocal = pb.multiplyTransforms(
            positionA=worldTransCom[0],
            orientationA=worldTransCom[1],
            positionB=comTransLocal[0],
            orientationB=comTransLocal[1],
            physicsClientId=self.cid,
        )
        return np.array(worldTransLocal[0])

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

        NOTE: Not to be used in high-speed loops as this operation can be heavy.
        """
        kwargs.pop("bodyUniqueId", None)
        kwargs.pop("linkIndex", None)
        kwargs.pop("physicsClientId", None)
        pb.changeDynamics(
            bodyUniqueId=self.robot_id,
            linkIndex=self.get_link_index(link_name=link_name),
            physicsClientId=self.cid,
            **kwargs,
        )
        self.refresh_sim_data()

    # Get actuated joint states
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

    def get_actuated_joint_name_to_position(self) -> Dict[str, float]:
        return dict(
            zip(
                self.actuated_joint_names,
                self.get_joint_states(joint_ids=self.actuated_joint_ids)[0].tolist(),
            )
        )

    def get_actuated_joint_name_to_velocity(self) -> Dict[str, float]:
        return dict(
            zip(
                self.actuated_joint_names,
                self.get_joint_states(joint_ids=self.actuated_joint_ids)[1].tolist(),
            )
        )

    def get_actuated_joint_name_to_torque(self) -> Dict[str, float]:
        return dict(
            zip(
                self.actuated_joint_names,
                self.get_joint_states(joint_ids=self.actuated_joint_ids)[2].tolist(),
            )
        )

    def get_jacobian(self, ee_link_name: str, joint_angles=None) -> np.ndarray:
        if joint_angles is None:
            joint_angles = self.get_actuated_joint_positions()

        linear_jacobian, angular_jacobian = pb.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.get_link_index(ee_link_name),
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
            link_ids=[self.get_link_index(link_name=name) for name in ee_names]
        )

    def get_contact_states_of_links(self, link_ids: List[int]) -> np.ndarray:
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

    # Get robot states
    def get_robot_states(
        self, actuated_joint_names: List[str] = None, ee_names: List[str] = None
    ) -> Dict[str, Any]:
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
            joint_ids = self.actuated_joint_ids
        else:
            joint_ids = self.get_joint_ids(joint_names=actuated_joint_names)
        if ee_names is None:
            ee_ids = self.ee_ids
            ee_names = self.ee_names
        else:
            ee_ids = [self.get_link_index(link_name=name) for name in ee_names]

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
        if joint_ids is None:
            joint_ids = self.actuated_joint_ids
        dof = len(joint_ids)
        q, v, tau = np.zeros(dof), np.zeros(dof), np.zeros(dof)
        for n, idx in enumerate(joint_ids):
            q[n], v[n], _, tau[n] = pb.getJointState(
                self.robot_id, idx, physicsClientId=self.cid
            )
            if idx in self.continuous_joint_ids:
                q[n] = wrap_angle(q[n])
            if self._in_torque_mode:
                tau[n] = self._prev_torque_cmds[idx]
        return q, v, tau

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
        if actuated_joint_names is None:
            actuated_joint_names = self.actuated_joint_names
        jids = self.get_joint_ids(actuated_joint_names)

        pb.setJointMotorControlArray(
            self.robot_id,
            jids,
            controlMode=pb.POSITION_CONTROL,
            targetPositions=cmd,
            targetVelocities=vels,
            physicsClientId=self.cid,
        )

    def set_actuated_joint_commands(
        self,
        actuated_joint_names: List[str] = None,
        q: float | np.ndarray = 0,
        Kp: float | np.ndarray = 0,
        dq: float | np.ndarray = 0,
        Kd: float | np.ndarray = 0,
        tau: float | np.ndarray = 0,
    ):
        # can be further optimised for speed by storing last used actuated joint names, e.g.
        # if using in a control loop

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
