import numpy as np
import pybullet as pb
from pybullet_robot.worlds import SimpleWorld
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSImpedanceController
import os, time

table_path = os.path.dirname(os.path.abspath(__file__)) + \
    '/../src/pybullet_robot/worlds/models/table.urdf'
plane_path = os.path.dirname(os.path.abspath(__file__)) + \
    '/../src/pybullet_robot/worlds/models/plane.urdf'

if __name__ == "__main__":
    robot = PandaArm()
    plane = pb.loadURDF(plane_path)
    table = pb.loadURDF(table_path, useFixedBase=True, globalScaling=0.5)
    pb.resetBasePositionAndOrientation(
        table, [0.6, 0., 0.3], [0, 0, -0.707, 0.707])

    objects = {'plane': plane,
               'table': table}

    world = SimpleWorld(robot, objects)

    slow_rate = 100.

    goal_pos, goal_ori = world.robot.ee_pose()

    controller = OSImpedanceController(robot)

    print "started"

    z_traj = np.linspace(goal_pos[2], 0.34, 500)

    controller.start_controller_thread()

    i = 0

    while i < z_traj.size:
        now = time.time()

        ee_pos, _ = world.robot.ee_pose()
        wrench = world.robot.get_ee_wrench(local=False)
        
        goal_pos[2] = z_traj[i]

        print "Goal:", ee_pos, "Actual:", goal_pos
        controller.update_goal(goal_pos, goal_ori)

        elapsed = time.time() - now
        sleep_time = (1./slow_rate) - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        i+=1
            
    controller.stop_controller_thread()
