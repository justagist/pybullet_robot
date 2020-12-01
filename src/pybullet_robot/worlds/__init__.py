from simple_world import SimpleWorld


def add_PbR_models_to_path():
    """
    adds the models in Pybullet Robot world module to the
    pybullet path for easily retrieving the models
    """
    import pybullet as pb
    import os
    # pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(os.path.dirname(
        os.path.abspath(__file__)) + '/models')

    # pb.resetSimulation()


def add_PyB_models_to_path():
    """
    adds pybullet's in-built models path to the
    pybullet path for easily retrieving the models
    """
    import pybullet as pb
    import pybullet_data
    # pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # pb.resetSimulation()

