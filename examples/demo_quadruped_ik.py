"""Whole-body (multi-end-effector) inverse-kinematics demo on a quadruped.

Loads a Unitree A1 with a FLOATING base and tracks five frames at once with
``PybulletIKInterface``: the body pose plus all four feet. The four feet are held planted on
the ground (green markers) while you drag the "body height" slider; the IK interface solves
the floating base pose together with all twelve leg joints so the body squats and rises while
the feet stay put.

This showcases the headline feature of the IK interface: simultaneous multi-end-effector,
floating-base IK solved purely by forward-simulating physics constraints.

Run:
    python demo_quadruped_ik.py
"""

import time

import numpy as np
import pybullet as pb

from pybullet_robot import PybulletIKInterface
from pybullet_robot.utils.robot_loader_utils import (
    get_urdf_from_awesome_robot_descriptions,
)

FEET = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
STAND_HEIGHT = 0.3


def main():
    ik = PybulletIKInterface(
        urdf_path=get_urdf_from_awesome_robot_descriptions("a1_description"),
        floating_base=True,
        starting_base_position=[0, 0, STAND_HEIGHT],
        frames_to_track_pose=["trunk"],  # body pose task
        frames_to_track_position=FEET,  # one point task per foot
        visualise=True,
        run_async=True,
        update_rate=500,
        disable_gravity=True,
    )
    cid = ik.cid
    time.sleep(0.6)  # let the solver settle into the standing posture

    body_home, body_ori = ik.get_frame_target("trunk")
    foot_home = {f: ik.get_point_target(f) for f in FEET}
    # set all four feet on the ground
    for f in FEET:
        foot_home[f][2] = 0.0
        ik.update_point_task(f, target_position=foot_home[f])

    # Green markers at each tracked foot target (they should stay planted).
    foot_marker_vis = pb.createVisualShape(
        pb.GEOM_SPHERE, radius=0.025, rgbaColor=[0, 1, 0, 0.9], physicsClientId=cid
    )
    for f in FEET:
        pb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=foot_marker_vis,
            basePosition=foot_home[f],
            physicsClientId=cid,
        )

    # Slider to set the body height (drag down to squat). Range stays within the leg workspace.
    height_slider = pb.addUserDebugParameter(
        "body height",
        STAND_HEIGHT - 0.2,
        STAND_HEIGHT,
        STAND_HEIGHT,
        physicsClientId=cid,
    )
    body_x_slider = pb.addUserDebugParameter(
        "body x",
        -0.1,
        0.1,
        0.0,
        physicsClientId=cid,
    )
    body_y_slider = pb.addUserDebugParameter(
        "body y",
        -0.1,
        0.1,
        0.0,
        physicsClientId=cid,
    )

    print("\nDrag the 'body height' slider to squat/stand. Press Ctrl+C to exit.\n")
    last_log = 0.0
    try:
        while True:
            height = pb.readUserDebugParameter(height_slider, physicsClientId=cid)
            body_x = pb.readUserDebugParameter(body_x_slider, physicsClientId=cid)
            body_y = pb.readUserDebugParameter(body_y_slider, physicsClientId=cid)
            body_target = np.array([body_x, body_y, height])
            ik.update_frame_task(
                "trunk", target_position=body_target, target_orientation=body_ori
            )

            solution = ik.get_ik_solution()
            foot_err = max(e[0] for e in ik.compute_point_tracking_errors().values())

            now = time.time()
            if now - last_log > 1.0:
                base_xyz = solution.q[:3]
                leg_joints_deg = np.round(np.degrees(solution.q[7:]), 1)
                print(
                    f"body height cmd={height:.3f} m | solved base={np.round(base_xyz, 3)} "
                    f"| max foot error={foot_err * 1000:4.1f} mm | leg joints (deg)={leg_joints_deg}"
                )
                last_log = now

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        ik.close()


if __name__ == "__main__":
    main()
