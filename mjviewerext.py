__author__ = 'DafniAntotsiou'

from mujoco_py.builder import functions
from mujoco_py import MjViewer
from mujoco_py.generated import const
import glfw
from copy import deepcopy
import numpy as np
from datetime import datetime
from gym.utils.seeding import np_random
import cv2


#extends the MjViewer class
class MjViewerExt(MjViewer):

    def __init__(self, sim):
        self._record_obs = False
        self._obs_idx = 1
        self.fps = 60
        self._obs_process = None
        self._prev_time = None

        self._trajectory = None
        self.rand_init = True
        self.np_random, seed = np_random(None)

        self.hpe = None     # record hpe skeleton
        self.img = None     # record hpe image
        self._record_path = None  # record path for recording programmatically

        super().__init__(sim)

    def _create_full_overlay(self):
        super()._create_full_overlay()
        self.add_overlay(const.GRID_TOPLEFT, "Reset", "[Backspace]")
        self.add_overlay(const.GRID_TOPLEFT, "Save Frame", "[F4]")
        self.add_overlay(const.GRID_TOPLEFT, "Record trajectories[F5]", "Disabled" if self._record_path is not None else ("On" if self._record_obs else "Off"))

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            if key == glfw.KEY_BACKSPACE:
                if self._record_obs:
                    self.__save_obs()

                # reset
                functions.mj_resetData(self.sim.model, self.sim.data)
                if self.rand_init:
                    addr = self.sim.model.get_joint_qpos_addr('forearm')[0]
                    self.sim.data.qpos[addr + 3: addr + 7] = np.array([0, 0, 0, 1])

                    addr = self.sim.model.get_joint_qpos_addr('Object')[0]
                    obj_height = self.sim.data.qpos[addr + 2]
                    self.sim.data.qpos[:] += self.np_random.uniform(low=-.1, high=.1, size=self.sim.model.nq)
                    self.sim.data.qpos[addr + 2] = obj_height

                    self.sim.data.set_mocap_pos('mocap', self.sim.data.get_joint_qpos('forearm')[0:3])
                    self.sim.data.set_mocap_quat('mocap', self.sim.data.get_joint_qpos('forearm')[3:7])


            elif key == glfw.KEY_F5 and self._record_path is None:
                self._record_obs = not self._record_obs
                # record states and actions
                if not self._record_obs:
                    # record turned off - save current trajectory
                    self.__save_obs()

            elif key == glfw.KEY_F4:
                # save current frame
                mocap = np.concatenate((self.sim.data.mocap_pos.flatten(), self.sim.data.mocap_quat.flatten()), axis=0)
                frame = {'obs': np.concatenate((np.asarray(self.sim.data.qpos),
                                                               np.asarray(self.sim.data.qvel)), axis=0),
                         'acs': np.concatenate((np.asarray(mocap), np.asarray(self.sim.data.ctrl)), axis=0),
                         'hpe': np.array(self.hpe)}
                name = 'frame_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S")
                np.savez(name, **frame)
                if self.img is not None:
                    cv2.imwrite(name + '.jpg', self.img)

        super().key_callback(window, key, scancode, action, mods)

    def render(self):
        if self._record_obs:
            if self._prev_time is None or self.sim.data.time - self._prev_time >= 1 / self.fps:
                # save frame
                self.save_frame()
                self._prev_time = deepcopy(self.sim.data.time)

        super().render()

    def start_rec(self, record_path):
        self.stop_rec()      # save trajectory if it's currently recording and start a new one

        self._record_path = record_path
        self._record_obs = True

    def stop_rec(self):
        if self._record_obs:
            self.__save_obs()
            self._record_path = None
            self._record_obs = False

    def __save_obs(self):
        if self._trajectory is not None and self._trajectory['obs'] and self._trajectory['acs']:
            #name = 'trajectory_' + str(self._obs_idx)
            name = 'trajectory_' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") if self._record_path is None else self._record_path

            for key in self._trajectory:
                self._trajectory[key] = np.array(self._trajectory[key])

            np.savez(name, **self._trajectory)

            self._obs_idx += 1

        self._trajectory = None
        self._prev_time = None

    def save_frame(self):
        if self._trajectory is None:
            self._trajectory = {'obs': [], 'acs': [], 'hpe': []}

        self._trajectory['obs'].append(np.concatenate((np.asarray(self.sim.data.qpos),
                                                       np.asarray(self.sim.data.qvel)), axis=0))
        mocap = np.concatenate((self.sim.data.mocap_pos.flatten(), self.sim.data.mocap_quat.flatten()), axis=0)
        self._trajectory['acs'].append(np.concatenate((np.asarray(mocap), np.asarray(self.sim.data.ctrl)), axis=0))
        if self.hpe is not None:
            self._trajectory['hpe'].append(np.array(self.hpe))

    def is_recording(self):
        return self._record_obs
