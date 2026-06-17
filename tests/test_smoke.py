"""Smoke tests for pybullet_robot."""

import os

import numpy as np
import pybullet as pb
import pybullet_data
import pytest

import pybullet_robot
from pybullet_robot import BulletRobot, PybulletIKInterface

KUKA_URDF = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa", "model.urdf")
NUM_JOINTS = 7  # KUKA iiwa has 7 actuated joints


def test_public_api():
    assert pybullet_robot.__all__ == [
        "BulletRobot",
        "PybulletIKInterface",
        "__version__",
    ]
    assert isinstance(pybullet_robot.__version__, str)
    assert BulletRobot is pybullet_robot.BulletRobot
    assert PybulletIKInterface is pybullet_robot.PybulletIKInterface


@pytest.fixture
def direct_client():
    cid = pb.connect(pb.DIRECT)
    yield cid
    pb.disconnect(cid)


def test_bullet_robot_basics(direct_client):
    robot = BulletRobot(
        urdf_path=KUKA_URDF,
        cid=direct_client,
        run_async=False,
        place_on_ground=False,
        load_ground_plane=False,
        verbose=False,
    )

    assert robot.num_actuated_joints == NUM_JOINTS
    assert len(robot.actuated_joint_names) == NUM_JOINTS
    assert len(robot.actuated_joint_ids) == NUM_JOINTS

    q, v, tau = robot.get_joint_states()
    assert q.shape == (NUM_JOINTS,)
    assert v.shape == (NUM_JOINTS,)
    assert tau.shape == (NUM_JOINTS,)

    grav = robot.get_gravity_compensation_torques()
    assert grav.shape == (NUM_JOINTS,)
    assert np.all(np.isfinite(grav))

    jac = robot.get_jacobian(robot.link_names[-1])
    assert jac.shape == (6, NUM_JOINTS)

    state = robot.get_robot_states()
    for key in (
        "base_position",
        "base_quaternion",
        "actuated_joint_positions",
        "actuated_joint_velocities",
    ):
        assert key in state


def test_bullet_robot_reset_and_step(direct_client):
    robot = BulletRobot(
        urdf_path=KUKA_URDF,
        cid=direct_client,
        run_async=False,
        place_on_ground=False,
        load_ground_plane=False,
        verbose=False,
    )
    target = np.full(robot.num_actuated_joints, 0.3)
    robot.reset_actuated_joint_positions(joint_positions=target)
    q, _, _ = robot.get_joint_states()
    assert np.allclose(q, target, atol=1e-3)
    robot.step()  # should be a no-op-safe call in sync mode


def test_ik_interface_solution_shape():
    ik = PybulletIKInterface(
        urdf_path=KUKA_URDF,
        floating_base=False,
        run_async=False,
        visualise=False,
    )
    try:
        solution = ik.get_ik_solution()
        assert solution.success
        assert solution.q.shape == (NUM_JOINTS,)
        assert solution.v.shape == (NUM_JOINTS,)
    finally:
        ik.close()
