"""Headless IK solving + control in a separate pybullet instance.

This demonstrates a realistic split between *solving* IK and *controlling* a robot:

1. ``PybulletIKInterface`` runs **headless** (``visualise=False`` -> a DIRECT pybullet client)
   and is used purely as an "IK solver". It owns its own physics client and constraint world.
2. A **separate** pybullet GUI client holds the robot we actually control. This is used to demonstrate
   the control loop running on a real robot or a simulated robot. Each loop we read a target end-effector pose
   from the GUI sliders, ask the headless solver for the joint solution, and command that robot with position control.

The two simulations are completely independent (different physics client ids); only the joint
solution vector is passed from the solver to the controlled robot.

Run:
    python demo_ik_headless_control.py
"""

import time

import numpy as np
import pybullet as pb

from pybullet_robot import PybulletIKInterface
from pybullet_robot.bullet_robot import BulletRobot
from pybullet_robot.utils.robot_loader_utils import (
    get_urdf_from_awesome_robot_descriptions,
)

from demo_utils import add_ground, prettify_gui

EE_LINK = "iiwa_link_ee"


def main():
    urdf = get_urdf_from_awesome_robot_descriptions("iiwa14_description")

    # 1) IK SOLVER -- headless (no GUI). Solves IK by forward-simulating constraints in DIRECT.
    ik = PybulletIKInterface(
        urdf_path=urdf,
        floating_base=False,
        frames_to_track_pose=[EE_LINK],
        visualise=False,
        run_async=True,
        update_rate=500,
    )
    time.sleep(0.5)  # let the solver settle at the home pose
    home_pos, home_ori = ik.get_frame_target(EE_LINK)

    # 2) CONTROLLED ROBOT -- in a SEPARATE pybullet GUI client.
    gui_cid = pb.connect(pb.GUI)
    robot = BulletRobot(
        urdf_path=urdf,
        cid=gui_cid,
        ee_names=[EE_LINK],
        run_async=False,
        place_on_ground=False,
        load_ground_plane=False,  # we add a clean ground below
        verbose=False,
    )
    robot.set_position_control_mode()
    add_ground(gui_cid)
    prettify_gui(gui_cid, camera_distance=2.0, camera_target=(0.0, 0.0, 0.8))
    print(
        f"\nIK solver client id: {ik.cid} (headless) | controlled robot client id: {robot.cid} (GUI)"
    )

    # Sliders (in the GUI client) to set the target end-effector position; orientation is held
    # at the home orientation.
    sliders = {
        "x": pb.addUserDebugParameter(
            "target x",
            home_pos[0] - 0.6,
            home_pos[0] + 0.6,
            home_pos[0],
            physicsClientId=gui_cid,
        ),
        "y": pb.addUserDebugParameter(
            "target y",
            home_pos[1] - 0.6,
            home_pos[1] + 0.6,
            home_pos[1],
            physicsClientId=gui_cid,
        ),
        "z": pb.addUserDebugParameter(
            "target z",
            max(0.05, home_pos[2] - 0.7),
            home_pos[2] + 0.2,
            home_pos[2],
            physicsClientId=gui_cid,
        ),
    }
    marker_vis = pb.createVisualShape(
        pb.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8], physicsClientId=gui_cid
    )
    marker = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=marker_vis,
        basePosition=home_pos,
        physicsClientId=gui_cid,
    )

    print(
        "Drag the sliders in the GUI to set a target. The headless solver computes joint"
    )
    print("angles and the GUI robot is commanded to them. Press Ctrl+C to exit.\n")
    last_log = 0.0
    try:
        while True:
            target_pos = np.array(
                [
                    pb.readUserDebugParameter(sliders[k], physicsClientId=gui_cid)
                    for k in ("x", "y", "z")
                ]
            )

            # --- solve IK headless ---
            ik.update_frame_task(
                EE_LINK, target_position=target_pos, target_orientation=home_ori
            )
            solution = ik.get_ik_solution()

            # --- apply the solution to the robot in the separate GUI sim ---
            robot.set_joint_positions(cmd=solution.q)
            robot.step()
            pb.resetBasePositionAndOrientation(
                marker, target_pos, [0, 0, 0, 1], physicsClientId=gui_cid
            )

            now = time.time()
            if now - last_log > 1.0:
                ee_actual = robot.get_link_pose(robot.ee_ids[0])[0]
                print(
                    f"target={np.round(target_pos, 3)} "
                    f"| IK q (deg)={np.round(np.degrees(solution.q), 1)} "
                    f"| controlled-robot EE={np.round(ee_actual, 3)}"
                )
                last_log = now

            time.sleep(1.0 / 240.0)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        ik.close()
        robot.shutdown()


if __name__ == "__main__":
    main()
