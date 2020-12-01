import numpy as np
import pybullet as pb
from pybullet_robot.worlds import SimpleWorld, add_PyB_models_to_path
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSImpedanceController
import time

if __name__ == "__main__":
    robot = PandaArm()

    add_PyB_models_to_path()

    plane = pb.loadURDF('plane.urdf')
    table = pb.loadURDF('table/table.urdf',
                        useFixedBase=True, globalScaling=0.5)
    cube = pb.loadURDF('cube_small.urdf', useFixedBase=True, globalScaling=1.)
    pb.resetBasePositionAndOrientation(
        table, [0.4, 0., 0.0], [0, 0, -0.707, 0.707])

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
