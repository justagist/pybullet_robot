"""Small shared helpers to make the example pybullet scenes look nicer.

These are purely cosmetic (camera, lighting, a clean ground) and are not part of the package
itself; they just keep the demos visually tidy.
"""

import pybullet as pb

GROUND_COLOR = [0.85, 0.87, 0.90, 1.0]


def prettify_gui(
    cid,
    camera_distance=1.8,
    camera_yaw=50.0,
    camera_pitch=-30.0,
    camera_target=(0.0, 0.0, 0.4),
    keep_panel=True,
):
    """Tidy up a pybullet GUI.

    Hides the RGB/depth/segmentation buffer-preview thumbnails, enables shadows, and points the
    camera at the scene. Set ``keep_panel=False`` to also hide the side panel (do NOT do this in
    demos that rely on debug-parameter sliders, since the sliders live in that panel).
    """
    for flag in (
        pb.COV_ENABLE_RGB_BUFFER_PREVIEW,
        pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
        pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
    ):
        pb.configureDebugVisualizer(flag, 0, physicsClientId=cid)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1, physicsClientId=cid)
    if not keep_panel:
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0, physicsClientId=cid)
    pb.resetDebugVisualizerCamera(
        camera_distance,
        camera_yaw,
        camera_pitch,
        list(camera_target),
        physicsClientId=cid,
    )


def add_ground(
    cid, z=0.0, half_extents=(3.0, 3.0, 0.005), color=GROUND_COLOR, collision=False
):
    """Add a clean, solid-coloured ground plane to get rid of the ugly checkerboard grid."""
    half_extents = list(half_extents)
    visual = pb.createVisualShape(
        pb.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, physicsClientId=cid
    )
    collision_idx = (
        pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=half_extents, physicsClientId=cid
        )
        if collision
        else -1
    )
    return pb.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual,
        baseCollisionShapeIndex=collision_idx,
        basePosition=[0, 0, z - half_extents[2]],
        physicsClientId=cid,
    )
