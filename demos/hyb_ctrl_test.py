from collections import deque
import numpy as np
import pybullet as pb
from pybullet_robot.worlds import SimpleWorld
from pybullet_robot.robots import PandaArm
from pybullet_robot.controllers import OSHybridController
import os
import time
import matplotlib.pyplot as plt
import threading

table_path = os.path.dirname(os.path.abspath(__file__)) + \
    '/../src/pybullet_robot/worlds/models/table.urdf'
plane_path = os.path.dirname(os.path.abspath(__file__)) + \
    '/../src/pybullet_robot/worlds/models/plane.urdf'


def plot_thread():

    plt.ion()
    while True:
        plt.clf()
        # plt.subplot(, 1, 1)
        # plt.plot(fx_deque, 'r', label='x')
        # plt.ylim([0.3, 0.6])
        # plt.subplot(2, 1, 2)
        plt.plot(fx_deque, 'r', label='x')
        plt.plot(fy_deque, 'g', label='y')
        # plt.ylim([-0.15, 0])
        plt.plot(fz_deque, 'b', label='z')
        plt.legend()
        plt.draw()
        plt.pause(0.000001)
        if done:
            break

if __name__ == "__main__":
    robot = PandaArm()
    plane = pb.loadURDF(plane_path)
    table = pb.loadURDF(table_path, useFixedBase=True, globalScaling=0.5)
    pb.resetBasePositionAndOrientation(
        table, [0.6, 0., 0.3], [0, 0, -0.707, 0.707])

    objects = {'plane': plane,
               'table': table}

    world = SimpleWorld(robot, objects)
    pb.changeDynamics(world.objects.table, -1,
                      lateralFriction=0.1, restitution=0.9)
    slow_rate = 100.

    goal_pos, goal_ori = world.robot.ee_pose()

    controller = OSHybridController(robot)

    print ("started")

    z_traj = np.linspace(goal_pos[2], 0.3, 550)

    plot_t = threading.Thread(target=plot_thread)
    fx_deque = deque([0],maxlen=1000)
    fy_deque = deque([0],maxlen=1000)
    fz_deque = deque([0],maxlen=1000)

    controller.start_controller_thread()

    done = False
    plot_t.start()

    i = 0
    f_ctrl = True
    while i < z_traj.size:
        now = time.time()

        ee_pos, _ = world.robot.ee_pose()
        wrench = world.robot.get_ee_wrench(local=False)
        # print wrench
        if abs(wrench[2]) >= 10.:
            break

        goal_pos[2] = z_traj[i]

        controller.update_goal(goal_pos, goal_ori)

        fx_deque.append(wrench[0])
        fy_deque.append(wrench[1])
        fz_deque.append(wrench[2])
        
        elapsed = time.time() - now
        sleep_time = (1./slow_rate) - elapsed
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        i += 1
    else:
        print ("Never reached force threshold for switching controller")
        f_ctrl = False
    
    if f_ctrl:
    
        print ("Switching to force control along Z axis")
        y_traj = np.linspace(goal_pos[1], goal_pos[1]-0.2, 400)
        
        controller.change_ft_directions([0,0,1,0,0,0])
        target_force = -21.

        p_slider = pb.addUserDebugParameter('p_f',0.1,2.,controller._P_ft[2, 2])
        i_slider = pb.addUserDebugParameter('i_f',0.0,100.,controller._I_ft[2, 2])
        w_slider = pb.addUserDebugParameter('windup',0.0,100.,controller._windup_guard[2, 0])

        
        i = 0
        while i < y_traj.size:
            now = time.time()

            ee_pos, _ = world.robot.ee_pose()
            wrench = world.robot.get_ee_wrench(local=False)
            # print wrench
            goal_pos[1] = y_traj[i]

            controller._P_ft[2, 2] = pb.readUserDebugParameter(p_slider)
            controller._I_ft[2, 2] = pb.readUserDebugParameter(i_slider)
            controller._windup_guard[2, 0] = pb.readUserDebugParameter(w_slider)

            controller.update_goal(
                goal_pos, goal_ori, np.asarray([0., 0., target_force]))

            fx_deque.append(wrench[0])
            fy_deque.append(wrench[1])
            fz_deque.append(wrench[2])

            elapsed = time.time() - now
            sleep_time = (1./slow_rate) - elapsed
            if sleep_time > 0.0:
                time.sleep(sleep_time)

            if i < y_traj.size-1:
                i += 1

    controller.stop_controller_thread()
    done = True
    plot_t.join()
