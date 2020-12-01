import threading
import numpy as np
import time
import pybullet as pb


class OSControllerBase(object):

    def __init__(self, robot, config, break_condition=None):

        self._robot = robot # robot sim should not be in real-time. Step simulation will be called by controller.
        self._P_pos = np.diag(config['P_pos'])
        self._D_pos = np.diag(config['D_pos'])

        self._P_ori = np.diag(config['P_ori'])
        self._D_ori = np.diag(config['D_ori'])


        self._error_thresh = config['error_thresh']
        self._start_err = config['start_err']

        self._robot.set_ctrl_mode('tor')

        self._run_ctrl = False

        if break_condition is not None and callable(break_condition):
            self._break_condition = break_condition
        else:
            self._break_condition = lambda: False

        self._ctrl_thread = threading.Thread(target=self._control_thread)
        self._mutex = threading.Lock()

        self._sim_timestep = pb.getPhysicsEngineParameters()['fixedTimeStep']
        self._sim_time = 0.0

        if 'rate' not in config:
            self._ctrl_rate = 1./self._sim_timestep
        else:
            self._ctrl_rate = float(config['rate'])

    def update_goal(self):
        """
        Has to be implemented in the inherited class.
        Should update the values for self._goal_pos and self._goal_ori at least.
        """
        raise NotImplementedError("Not implemented")

    def update_goal(self):
        """
        Has to be implemented in the inherited class.
        Should update the values for self._goal_pos and self._goal_ori at least.
        """
        raise NotImplementedError("Not implemented")

    def _compute_cmd(self):
        """
        Should be implemented in inherited class. Should compute the joint torques
        that are to be applied at every sim step.
        """
        raise NotImplementedError("Not implemented")

    def _control_thread(self):
        """
            Apply the torque command computed in _compute_cmd until any of the 
            break conditions are met.
        """
        while self._run_ctrl and not self._break_condition():
            error = self._start_err
            while np.any(error > self._error_thresh):
                now = time.time()
                
                self._mutex.acquire()
                tau, error = self._compute_cmd()
                
                # command robot using the computed joint torques
                self._robot.exec_torque_cmd(tau)

                self._robot.step_if_not_rtsim()
                self._sim_time += self._sim_timestep
                self._mutex.release()

                # self._rate.sleep()
                elapsed = time.time() - now
                sleep_time = (1./self._ctrl_rate) - elapsed
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    def _initialise_goal(self):
        """
        Should initialise _goal_pos, _goal_ori, etc. for controller to start the loop.
        Ideally these should be the current value of the robot's end-effector.
        """
        raise NotImplementedError("Not implemented")

    def start_controller_thread(self):
        self._initialise_goal()
        self._run_ctrl = True
        self._ctrl_thread.start()

    def stop_controller_thread(self):
        self._run_ctrl = False
        if self._ctrl_thread.is_alive():
            self._ctrl_thread.join()

    def __del__(self):
        self.stop_controller_thread()

