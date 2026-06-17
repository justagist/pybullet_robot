"""Interactive inverse-kinematics demo (single end-effector).

Loads a KUKA iiwa14 and uses ``PybulletIKInterface`` to find joint angles that place the
end-effector at a target pose. Drag the sliders in the pybullet GUI to move the target (a red
marker); whenever the target changes, the IK interface forward-simulates its internal
constraints and converges to a joint configuration that reaches it.

This demonstrates that the package can *solve* IK for arbitrary reachable targets; each target is
solved once it is set.

For clarity this demo visualises the solver's *own* internal simulation (``visualise=True``),
i.e. you see the constraint-driven robot inside the IK interface itself. In a real application
you typically run the IK interface headless and apply its joint solution to the robot you
actually control (in your own simulation or on hardware). See ``demo_ik_headless_control.py``
for that more typical usage.

Run:
    python demo_ik_interface.py
"""

import time

import numpy as np
import pybullet as pb

from pybullet_robot import PybulletIKInterface
from pybullet_robot.utils.robot_loader_utils import (
    get_urdf_from_awesome_robot_descriptions,
)

from demo_utils import add_ground, prettify_gui

EE_LINK = "iiwa_link_ee"


def main():
    ik = PybulletIKInterface(
        urdf_path=get_urdf_from_awesome_robot_descriptions("iiwa14_description"),
        floating_base=False,
        frames_to_track_pose=[EE_LINK],
        visualise=True,  # opens a pybullet GUI
        run_async=True,  # solve continuously in a background thread
        update_rate=500,
    )
    cid = ik.cid
    add_ground(cid)  # visual-only, won't interfere with the IK constraints
    prettify_gui(cid, camera_distance=2.0, camera_target=(0.0, 0.0, 0.8))
    time.sleep(0.5)  # let the background solver settle at the home pose

    home_pos, home_ori = ik.get_frame_target(EE_LINK)
    home_rpy = pb.getEulerFromQuaternion(home_ori)

    # GUI sliders to set the target end-effector pose (position + orientation).
    sliders = {
        "x": pb.addUserDebugParameter(
            "target x",
            home_pos[0] - 0.6,
            home_pos[0] + 0.6,
            home_pos[0],
            physicsClientId=cid,
        ),
        "y": pb.addUserDebugParameter(
            "target y",
            home_pos[1] - 0.6,
            home_pos[1] + 0.6,
            home_pos[1],
            physicsClientId=cid,
        ),
        "z": pb.addUserDebugParameter(
            "target z",
            max(0.05, home_pos[2] - 0.7),
            home_pos[2] + 0.3,
            home_pos[2],
            physicsClientId=cid,
        ),
        "roll": pb.addUserDebugParameter(
            "target roll", -np.pi, np.pi, home_rpy[0], physicsClientId=cid
        ),
        "pitch": pb.addUserDebugParameter(
            "target pitch", -np.pi, np.pi, home_rpy[1], physicsClientId=cid
        ),
        "yaw": pb.addUserDebugParameter(
            "target yaw", -np.pi, np.pi, home_rpy[2], physicsClientId=cid
        ),
    }

    # A red sphere marking the commanded target pose.
    marker_vis = pb.createVisualShape(
        pb.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0, 0, 0.8], physicsClientId=cid
    )
    marker = pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=marker_vis,
        basePosition=home_pos,
        physicsClientId=cid,
    )

    print(
        "\nDrag the sliders in the GUI to set a target EE pose. Press Ctrl+C to exit."
    )
    print(
        "(This shows the solver's own internal sim. For the typical usage, solving IK headless "
        "and controlling a robot in a separate sim, see demo_ik_headless_control.py.)\n"
    )
    prev_target = None
    last_log = 0.0
    try:
        while True:
            target_pos = np.array(
                [
                    pb.readUserDebugParameter(sliders[k], physicsClientId=cid)
                    for k in ("x", "y", "z")
                ]
            )
            target_rpy = [
                pb.readUserDebugParameter(sliders[k], physicsClientId=cid)
                for k in ("roll", "pitch", "yaw")
            ]
            target_ori = pb.getQuaternionFromEuler(target_rpy)

            ik.update_frame_task(
                EE_LINK, target_position=target_pos, target_orientation=target_ori
            )
            pb.resetBasePositionAndOrientation(
                marker, target_pos, [0, 0, 0, 1], physicsClientId=cid
            )

            # compute the IK solution for the current target
            solution = ik.get_ik_solution()
            # these joint solutions can now be used as the IK solution if the error is small enough

            error = ik.compute_frame_tracking_errors()[EE_LINK]

            # Log when the target moves or once a second.
            now = time.time()
            moved = (
                prev_target is None or np.linalg.norm(target_pos - prev_target) > 1e-3
            )
            if moved or now - last_log > 1.0:
                print(
                    f"target={np.round(target_pos, 3)} "
                    f"| IK joint solution (deg)={np.round(np.degrees(solution.q), 1)} "
                    f"| reached: {error[0] * 1000:5.1f} mm, {error[1]:.3f} rad"
                )
                prev_target = target_pos
                last_log = now

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        ik.close()


if __name__ == "__main__":
    main()
