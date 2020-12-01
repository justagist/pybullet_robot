import pybullet as pb
import numpy as np
import time


class WorldObjects(object):

    def __init__(self, object_dict={}):

        for k in object_dict:
            setattr(self, k, object_dict[k])

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)


class SimpleWorld(object):

    def __init__(self, robot, other_objects={}):

        self._robot = robot
        self._other_objects = WorldObjects(other_objects)

        self.add_objects({'robot': self._robot._id})

        self.reset_world()

    def reset_world(self, timeout=2):
        if self._robot._rt_sim:
            self._robot.untuck()
            return

        timeout = time.time() + timeout
        while time.time() < timeout:
            self._robot.untuck()
            pb.stepSimulation()

    def step(self):
        self._robot.step_sim()

    def add_objects(self, objects):

        for k in objects:
            print ("Adding", k)
            if not hasattr(self._other_objects, k):
                self._other_objects[k] = objects[k]
            else:
                print (
                    "Object with keyname '{}' already present. Not adding to world.".format(k))

    def remove_objects(self, objects=[]):
        if isinstance(objects, str):
            objects = [objects]
        for obj in objects:
            delattr(self._other_objects, obj)

    @property
    def objects(self):
        return self._other_objects

    @property
    def robot(self):
        return self._robot
