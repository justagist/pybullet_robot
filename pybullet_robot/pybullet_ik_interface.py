"""
Class that finds joint positions to achieve multi-end-effector targets provided,
using pybullet's internal physics simulation and constraints.
"""

from typing import List, Mapping, Tuple
from dataclasses import dataclass, field
import logging
import time
from threading import Thread, Lock
import atexit
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation

from .bullet_robot import BulletRobot, QuatType, Vector3D

# pylint: disable = I1101


def _quat_error(quat1: QuatType, quat2: QuatType) -> float:
    return (Rotation.from_quat(quat1) * Rotation.from_quat(quat2).inv()).magnitude()


def _transform_pose_to_frame(
    pos_in_frame_1: Vector3D,
    quat_in_frame_1: QuatType,
    frame_2_pos_in_frame_1: Vector3D,
    frame_2_quat_in_frame_1: QuatType,
) -> Tuple[Vector3D, QuatType]:
    p, q = pb.multiplyTransforms(
        *pb.invertTransform(frame_2_pos_in_frame_1, frame_2_quat_in_frame_1),
        pos_in_frame_1,
        quat_in_frame_1,
    )
    return np.array(p), np.array(q)


class PybulletIKInterface:
    """Compute joint positions for provided end-effector references for multi-end-effector robots
    using physics engine...

    This class doesn't actuatlly solve IK in the traditional sense. Instead each end-effector
    target is set to the desired pose in the physics world by attaching a constraint (using
    pybullet `createConstraint` API) to the physics. Then "IK solution" is queried by forward
    simulating the world with these constraints and querying the joint states of the robot. This
    formulation should, in theory, generalise to a robot with any number of end-effectors.

    Cons:
        - This runs a separate thread where physics is forward simulated at a higher frequency.
            - This means velocity solutions are not reliable (different time scale)!
            - All typical issues with using threads...
            - Solutions not necessarily repeatable!
        - No collision avoidance (in the traditional sense).
        - Perfect solution not guaranteed! (constraints may be infeasible, may cause physics
            inconsistencies)

    Pros:
        - All joint solutions provided are physically feasible by the robot (no self-penetration,
            no singularities, etc)
        - Solutions can be enforced to be smooth in joint space by changing joint friction etc.
        - Fast enough to be usable in "real-time" control loop.
        - Closest solution will be provided even if perfect tracking is not achievable..? (NOT SURE
            IF THIS CLAIM IS VALID)
    """

    @dataclass
    class Solution:
        """Output for the `get_ik_solution` method in PybulletIKInterface.

        Attributes:
            success (bool): whether IK solution is reliable (all frame tracking tasks succesful).
                Only meaningful if error thresholds are specified in the call to `get_ik_solution`
                method.
            q (np.ndarray): The output generalised coordinate state (position) of the robot.
            v (np.ndarray): The output generalised velocities of the robot (use with caution. See
                "Cons" section above.)
            message (str): Error message, if IK is not succesful.
        """

        success: bool
        q: np.ndarray
        v: np.ndarray
        message: str = field(default="")

    def __init__(
        self,
        urdf_path: str,
        floating_base: bool = False,
        frames_to_track_pose: List[str] = None,
        frames_to_track_position: List[str] = None,
        starting_base_position: Vector3D = np.zeros(3),
        starting_base_orientation: QuatType = np.array([0, 0, 0, 1]),
        starting_joint_positions: List[float] = None,
        joint_names_order: List[str] = None,
        run_async: bool = False,
        visualise: bool = False,
        disable_gravity: bool = False,
        update_rate: float = 500,
        cid: int = None,
    ):
        """Compute IK solutions for provided end-effector references for multi-end-effector robots
        using physics engine...

        This class doesn't actuatlly solve IK in the traditional sense. Instead each end-effector
        target is set to the desired pose in the physics world by attaching a constraint (using
        pybullet `createConstraint` API) to the physics. Then "IK solution" is queried by forward
        simulating the world with these constraints and querying the joint states of the robot.
        This formulation should, in theory, generalise to a robot with any number of end-effectors.

        Args:
            urdf_path (str): Path to urdf of the robot.
            floating_base (bool, optional): Whether the robot has fixed base or floating base.
                Defaults to False.
            frames_to_track_pose (List[str], optional): Names of links to be tracked using target
                pose (position and orientation). These can also be added later using
                `add_frame_task`. Defaults to None.
            frames_to_track_position (List[str], optional): Names of links to be tracked in
                position only. These can also be added later using `add_point_task`. Defaults
                to None.
            starting_base_position (Vector3D, optional): Starting base position of the robot.
                Defaults to np.zeros(3).
            starting_base_orientation (QuatType, optional): Starting base orientation of the robot.
                Defaults to np.array([0, 0, 0, 1]).
            starting_joint_positions (List[float], optional): Starting joint positions of the
                robot. Defaults to None.
            joint_names_order (List[str], optional): Order of joint names to be used. This will be
                the order of the output of the IK as well. Defaults to None (use default order from
                BulletRobot instance when loading this urdf).
            run_async (bool, optional): If True, will run step simulation for this simulation in a
                separete thread at the frequency specified in `update_rate`.
            visualise (bool, optional): If True, will create a GUI instance of the simulated world.
                NOTE: This cannot be done if there is another GUI instance of pybullet already
                running in the same process. Defaults to False.
            update_rate (float, optional): The rate (Hz) at which the physics simulation step
                should be called. Only valid if `run_async` is True. Defaults to 500. (uses system
                clock)
            cid (int, optional): Physics client id of pybullet physics engine (if already running).
                If None provided, will create a new physics GUI instance. Defaults to None.
        """
        if not run_async and visualise and cid is not None:
            self.cid = cid
        else:
            self.cid = pb.connect(pb.GUI if visualise else pb.DIRECT)

        self._robot = BulletRobot(
            urdf_path=urdf_path,
            cid=self.cid,
            run_async=False,
            load_ground_plane=False,
            ghost_mode=False,
            enable_torque_mode=False,
            place_on_ground=False,
            default_base_position=starting_base_position,
            default_base_orientation=starting_base_orientation,
            use_fixed_base=(not floating_base),
            verbose=False,
        )

        if disable_gravity:
            pb.setGravity(0, 0, 0, physicsClientId=self.cid)

        self.joint_name_order = (
            joint_names_order
            if joint_names_order is not None
            else self._robot.actuated_joint_names
        )
        self.default_joint_id_order = self._robot.get_joint_ids(self.joint_name_order)

        if starting_joint_positions is not None:
            self._robot.reset_joints(
                joint_positions=starting_joint_positions,
                joint_ids=self.default_joint_id_order,
            )

        self.floating_base = floating_base

        self._frame_constraints: Mapping[str, int] = {}
        self._point_constraints: Mapping[str, int] = {}
        self._frame_to_idx: Mapping[str, int] = {}

        self._async_mode = run_async
        self._sim_step_mutex = Lock()
        if frames_to_track_pose is not None:
            for frame in frames_to_track_pose:
                self.add_frame_task(frame_name=frame)

        if frames_to_track_position is not None:
            for frame in frames_to_track_position:
                self.add_point_task(frame_name=frame)

        if self._async_mode:
            self._thread_running = False

            def _update_thread(frequency: float):
                logging.info(
                    "%s: Starting IK simulation thread...", self.__class__.__name__
                )
                self._thread_running = True
                dt = 1.0 / frequency
                while self._thread_running:
                    self._robot.step()
                    time.sleep(dt)

            self._run_thread = Thread(target=_update_thread, args=[update_rate])
            self._run_thread.start()

        atexit.register(self.close)

    def add_frame_task(self, frame_name: str):
        """Add a frame constraint, i.e. end-effector/link whose target pose is
        to be set before solving IK.

        Args:
            frame_name (str): Name of the link.
        """
        assert (
            frame_name not in self._frame_constraints
        ), f"A frame task for {frame_name} already exists!"
        frame_id = self._robot.get_link_id(link_name=frame_name)
        frame_pose = self._robot.get_link_pose(link_id=frame_id)
        frame_com_pose = self._robot.get_link_com_pose(link_id=frame_id)
        rel_pos, rel_ori = _transform_pose_to_frame(
            pos_in_frame_1=frame_pose[0],
            quat_in_frame_1=frame_pose[1],
            frame_2_pos_in_frame_1=frame_com_pose[0],
            frame_2_quat_in_frame_1=frame_com_pose[1],
        )
        self._frame_constraints[frame_name] = pb.createConstraint(
            parentBodyUniqueId=self._robot.robot_id,
            parentLinkIndex=frame_id,
            childBodyUniqueId=-1,  # world
            childLinkIndex=-1,  # unused
            jointType=pb.JOINT_FIXED,
            jointAxis=[1, 1, 1],
            parentFramePosition=rel_pos,
            childFramePosition=frame_pose[0],
            parentFrameOrientation=rel_ori,
            childFrameOrientation=frame_pose[1],
            physicsClientId=self.cid,
        )
        if not self._async_mode:
            self._robot.step()
        self._frame_to_idx[frame_name] = frame_id

    def frame_task_exists(self, frame_name: str) -> bool:
        """Check if a frame tracking task already exists for the specified frame.

        Args:
            frame_name (str): Name of frame to check.

        Returns:
            bool: True if exists
        """
        return frame_name in self._frame_constraints

    def point_task_exists(self, frame_name: str) -> bool:
        """Check if point tracking task already exists for the specified frame.

        Args:
            frame_name (str): Name of frame to check.

        Returns:
            bool: True if exists
        """
        return frame_name in self._point_constraints

    def add_point_task(self, frame_name: str):
        """Add a point constraint, i.e. end-effector/link whose target position (only) is
        to be set before solving IK.

        Args:
            frame_name (str): Name of the link.
        """
        assert (
            frame_name not in self._point_constraints
        ), f"A frame task for {frame_name} already exists!"
        frame_id = self._robot.get_link_id(link_name=frame_name)
        frame_pose = self._robot.get_link_pose(link_id=frame_id)
        frame_com_pose = self._robot.get_link_com_pose(link_id=frame_id)
        rel_pos, rel_ori = _transform_pose_to_frame(
            pos_in_frame_1=frame_pose[0],
            quat_in_frame_1=frame_pose[1],
            frame_2_pos_in_frame_1=frame_com_pose[0],
            frame_2_quat_in_frame_1=frame_com_pose[1],
        )
        self._point_constraints[frame_name] = pb.createConstraint(
            parentBodyUniqueId=self._robot.robot_id,
            parentLinkIndex=frame_id,
            childBodyUniqueId=-1,  # world
            childLinkIndex=-1,  # unused
            jointType=pb.JOINT_POINT2POINT,
            jointAxis=[1, 1, 1],
            parentFramePosition=rel_pos,
            childFramePosition=frame_pose[0],
            parentFrameOrientation=rel_ori,
            childFrameOrientation=frame_pose[1],
            physicsClientId=self.cid,
        )
        if not self._async_mode:
            self._robot.step()
        self._frame_to_idx[frame_name] = frame_id

    def update_frame_task(
        self,
        frame_name: str,
        target_position: Vector3D,
        target_orientation: QuatType,
        max_force: float = None,
        erp: float = None,
    ):
        """Update the target pose for a frame (if defined already using `add_frame_task` or in
        the constructor).

        Args:
            frame_name (str): Name of the frame whose reference is to be changed.
            target_position (Vector3D): The desired position of the frame in the world.
            target_orientation (QuatType): The desired orientation (quaternion) in the world.
            max_force (float, optional): Max force this constraint is allowed to apply to pull
                the link. Defaults to None (use pybullet default).
            erp (float, optional): Error reduction parameter for this constraint. Defaults to None
                (use pybullet default).
        """
        kwargs = {}
        if max_force is not None:
            kwargs["maxForce"] = max_force
        if erp is not None:
            kwargs["erp"] = erp

        pb.changeConstraint(
            self._frame_constraints[frame_name],
            jointChildPivot=target_position,
            jointChildFrameOrientation=target_orientation,
            physicsClientId=self.cid,
            **kwargs,
        )
        if not self._async_mode:
            self._robot.step()

    def update_point_task(
        self,
        frame_name: str,
        target_position: Vector3D,
        max_force: float = None,
        erp: float = None,
    ):
        """Update the target position for a point task frame (if defined already using
        `add_point_task` or in the constructor).

        Args:
            frame_name (str): Name of the frame whose reference is to be changed.
            target_position (Vector3D): The desired position of the frame in the world.
            max_force (float, optional): Max force this constraint is allowed to apply to pull
                the link. Defaults to None (use pybullet default).
            erp (float, optional): Error reduction parameter for this constraint. Defaults to None
                (use pybullet default).
        """
        kwargs = {}
        if max_force is not None:
            kwargs["maxForce"] = max_force
        if erp is not None:
            kwargs["erp"] = erp

        pb.changeConstraint(
            self._point_constraints[frame_name],
            jointChildPivot=target_position,
            physicsClientId=self.cid,
            **kwargs,
        )
        if not self._async_mode:
            self._robot.step()

    def remove_frame_task(self, frame_name: str):
        """Remove a frame constraint.

        Args:
            frame_name (str): The name of the frame whose constraint is to be removed.
        """
        pb.removeConstraint(
            self._frame_constraints[frame_name], physicsClientId=self.cid
        )
        if not self._async_mode:
            self._robot.step()
        self._frame_constraints.pop(frame_name)

    def remove_point_task(self, frame_name: str):
        """Remove a point constraint.

        Args:
            frame_name (str): The name of the frame whose constraint is to be removed.
        """
        pb.removeConstraint(
            self._point_constraints[frame_name], physicsClientId=self.cid
        )
        if not self._async_mode:
            self._robot.step()
        self._point_constraints.pop(frame_name)

    def _get_constraint_target(self, constraint_id: int) -> Tuple[Vector3D, QuatType]:
        constraint_info = pb.getConstraintInfo(constraint_id, physicsClientId=self.cid)
        return np.array(constraint_info[7]), (
            None
            if constraint_info[4] == pb.JOINT_POINT2POINT
            else np.array(constraint_info[9])
        )

    def get_frame_target(self, frame_name: str) -> Tuple[Vector3D, QuatType]:
        """Get the currently set target pose for the specified frame.

        Args:
            frame_name (str): Name of the frame

        Returns:
            Tuple[Vector3D, QuatType]: Position and orientation (quaternion) target for this frame.
        """
        return self._get_constraint_target(
            constraint_id=self._frame_constraints[frame_name]
        )

    def get_point_target(self, frame_name: str) -> Vector3D:
        """Get the currently set target position for the specified frame (defined as point task).

        Args:
            frame_name (str): Name of the frame

        Returns:
            Vector3D: Position target set for this frame.
        """
        return self._get_constraint_target(
            constraint_id=self._point_constraints[frame_name]
        )[0]

    def compute_frame_tracking_errors(
        self, stop_simulation: bool = False
    ) -> Mapping[str, Tuple[float, float]]:
        """Compute the current tracking error for each of the frame tasks.

        Args:
            stop_simulation (bool, optional): If True, will pause the simulation
                to make sure no change happens during this function call. Defaults
                to False.

        Returns:
            Mapping[str, Tuple[float, float]]: Dictionary from frame name -> [position error,
                orientation error] for each frame defined with frame task. Orientation error
                is the shortest angle between the two rotations (in range [0, pi]).
        """
        if stop_simulation:
            self._sim_step_mutex.acquire()
        elif not self._async_mode:
            self._robot.step()
        errors = {}
        for fname, constraint_id in self._frame_constraints.items():
            p, o = self._robot.get_link_pose(link_id=self._frame_to_idx[fname])
            p_des, o_des = self._get_constraint_target(constraint_id=constraint_id)
            errors[fname] = [
                np.linalg.norm(p_des - p),
                _quat_error(quat1=o, quat2=o_des),
            ]
        if stop_simulation:
            self._sim_step_mutex.release()
        return errors

    def compute_point_tracking_errors(
        self, stop_simulation: bool = False
    ) -> Mapping[str, float]:
        """Compute the current tracking error for each of the point tasks.

        Args:
            stop_simulation (bool, optional): If True, will pause the simulation
                to make sure no change happens during this function call. Defaults
                to False.

        Returns:
            Mapping[str, float]: Dictionary from frame name -> position error for
                each frame defined with point tracking task.
        """
        if stop_simulation:
            self._sim_step_mutex.acquire()
        elif not self._async_mode:
            self._robot.step()
        errors = {}
        for fname, constraint_id in self._point_constraints.items():
            p, _ = self._robot.get_link_pose(link_id=self._frame_to_idx[fname])
            p_des, _ = self._get_constraint_target(constraint_id=constraint_id)
            errors[fname] = [
                np.linalg.norm(p_des - p),
            ]
        if stop_simulation:
            self._sim_step_mutex.release()
        return errors

    def get_ik_solution(
        self, pos_error_threshold: float = None, ori_error_threshold: float = None
    ) -> Solution:
        """Get the generalised coordinates (joint positions, and base pose if floating base)
        and generalised velocities for the robot with all the defined constraints in place.

        Args:
            pos_error_threshold (float, optional): Max L2 norm position error for the frame
                tracking tasks. Defaults to None (no error check done).
            ori_error_threshold (float, optional): Max error in the smallest angle between the
                desired frame orientation and current frame orientation for all the frames in
                the frame tasks defined. Defaults to None (no error check done).

        Returns:
            Solution: Output for the `get_ik_solution` method in PybulletIKInterface. It has the
                following attributes:
                    success (bool): whether IK solution is reliable (all frame tracking tasks
                        succesful). Only meaningful if error thresholds are specified in the call
                        to `get_ik_solution` method.
                    q (np.ndarray): The output generalised coordinate state (position) of the
                        robot. If floating base, this is concatenated base position, base
                        quaternion (x,y,z,w), joint positions. Otherwise, just joint positions.
                    v (np.ndarray): The output generalised velocities of the robot (use with
                        caution. See "Cons" section above.) message (str): Error message, if IK is
                        not succesful. If floating base, this is concatenated base linear velocity
                        (in base frame), base angular velocity (base frame), joint velocities.
                        Otherwise, just joint velocities.
        """
        with self._sim_step_mutex:
            if not self._async_mode:
                self._robot.step()
            joint_pos, joint_vel, _ = self._robot.get_joint_states(
                self.default_joint_id_order
            )
            if not self.floating_base:
                return PybulletIKInterface.Solution(
                    success=True, q=joint_pos, v=joint_vel
                )

            msg = ""
            success = True

            if pos_error_threshold is not None or ori_error_threshold is not None:
                frame_errors = self.compute_frame_tracking_errors(stop_simulation=False)
                if pos_error_threshold is not None:
                    for fname, f_error in frame_errors.items():
                        if f_error[0] > pos_error_threshold:
                            success = False
                            msg = (
                                "Position error threshold violated in frame tracking task"
                                f" for {fname}."
                            )
                            msg += f" Threshold: {pos_error_threshold}; Error: {f_error[0]}"
                            break
                    if success:
                        point_errors = self.compute_point_tracking_errors(
                            stop_simulation=False
                        )
                        for fname, p_error in point_errors.items():
                            if p_error > pos_error_threshold:
                                success = False
                                msg = (
                                    "Position error threshold violated in point tracking task"
                                    f" for {fname}."
                                )
                                msg += f" Threshold: {pos_error_threshold}; Error: {p_error}"
                                break
                if success and ori_error_threshold is not None:
                    for fname, f_error in frame_errors.items():
                        if f_error[1] > ori_error_threshold:
                            success = False
                            msg = (
                                "Orientation error threshold violated in frame tracking "
                                f"task for {fname}."
                            )
                            msg += f" Threshold: {ori_error_threshold}; Error: {f_error[1]}"
                            break

            q = np.zeros(len(self.default_joint_id_order) + 7)
            bp, bo = self._robot.get_base_pose()
            q[:3] = bp
            q[3:7] = bo
            q[7:] = joint_pos

            v = np.zeros(len(self.default_joint_id_order) + 6)
            blv, bav = self._robot.get_base_velocity()
            v[:3] = blv
            v[3:6] = bav
            v[6:] = joint_vel

        if not success:
            logging.warning("IK solution may not be valid. %s", msg)

        return PybulletIKInterface.Solution(success=success, q=q, v=v, message=msg)

    def close(self):
        """Gracefully shutdown the IK sim thread."""
        if self._async_mode:
            self._thread_running = False
            if self._run_thread.is_alive():
                logging.info(
                    "%s: Closing IK simulation thread...", self.__class__.__name__
                )
                self._run_thread.join()
                logging.info(
                    "%s: Succesfully closed IK simulation thread.",
                    self.__class__.__name__,
                )
        self._robot.shutdown()
