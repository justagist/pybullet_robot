from simple_world import SimpleWorld

def add_PyB_models_to_path():
    """
    adds pybullet's in-built models path to the
    pybullet path for easily retrieving the models
    """
    import pybullet as pb
    import pybullet_data
    # pb.connect(pb.GUI)
    print pybullet_data.getDataPath()
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # pb.resetSimulation()

