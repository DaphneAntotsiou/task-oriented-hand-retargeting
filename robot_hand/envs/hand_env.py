import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils, spaces, error,logger
from math import ceil
import os
from os import path
from skinematics import quat
from gym.envs.robotics import rotations
from mapping.functions import get_active_contacts_dist, get_pair_contacts, get_sensor_pos, \
    get_sensor_hpe_names, get_joint_qpos, vector_norm, get_hand_names, get_joint_qvel
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(e))
from copy import deepcopy
import mapping

class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    mocap_timestep = 0  # first is corrupted

    visual = False  # switch off for training
    _fps = 30
    max_timestep = 60
    print_msg = False
    relative_mocap = True   # relative movement of the mocap compared to its current position
    euler_angles = True     # wrist rotation action is done in euler xyz angles instead of quaternions
    slowmo = False          # run simulation in slow motion for debugging purposes...
    acs_norm = True         # normalise actions to [-1, 1]
    _epsilon = 1            # probability of choosing an initial position from the trajectories instead of random
    _sigma = 0.01            # max distance of random initialisation from the default position

    def __init__(self, model_path):

        # model_path = "MPL_Sphere.xml"
        #
        # # custom init
        # if model_path.startswith("/"):
        #     fullpath = model_path
        # else:
        #     fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)

        # this is based on mujoco_env!
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)
        self.model = mujoco_py.load_model_from_path(model_path)
        nsubstep = int(ceil(1/(self._fps * self.model.opt.timestep)))
        self.sim = mujoco_py.MjSim(self.model, nsubsteps=nsubstep)
        self._success = False
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}
        self.name = None

        self.frame_skip = 1

        self.mean_obs = None
        self.std_obs = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.contact_pairs = get_pair_contacts(self.model)

        # get object height address to keep it constant
        self.obj_height_addr = self.sim.model.get_joint_qpos_addr('Object')[0] + 2

        self.init_qpos = np.array([
            0.04837554573599772,
            -0.015168984826475402,
            0.15,
            0.17240389707893286,
            0.005433523752199522,
            -0.018547722830258313,
            -0.9848367149276283,
            -0.05103726010457265,
            -0.011682167664796513,
            -0.27128999351750593,
            0.9906626703594927,
            0.9024198564505892,
            0.059641568570250804,
            -0.013591194101868088,
            -0.0009648267086494113,
            0.4082643682485277,
            0.11289554118610776,
            0.23026232251527856,
            0.1681338249023414,
            0.37380624569544124,
            0.2907002517522892,
            0.1958775073108922,
            0.34043440319006824,
            0.24303886475096365,
            0.13401296685916114,
            0.30544837578463063,
            0.3394486963030385,
            0.34943395331014676,
            0.23618832285030206,
            0.3153444737449917,
            0.0,
            0.1,
            0.029632818157539823,
            1.0,
            0.0,
            0.0,
            0.0])

        self.init_qvel = np.array([
            0.09816235352041686,
            -0.10525255442555156,
            -0.32740712704829505,
            0.03500507237169123,
            0.7851760373806417,
            -0.7418122520786317,
            0.027945575293553182,
            0.16232298805917783,
            0.27176756889926174,
            -4.172744625789879,
            0.47874227731131147,
            0.07735172708303298,
            0.6763303471308884,
            0.027919217867560193,
            0.7163679756566792,
            0.8143253362714539,
            0.3705494465821854,
            0.0030154875883144157,
            1.1398108507472646,
            -0.12020799369108087,
            -0.5729019317242875,
            0.016521201298468707,
            4.863598408832601,
            -4.049481284998594,
            -4.307833064770609,
            0.02788325700913312,
            1.9947358295033066,
            -3.543801554892441,
            1.4702822126496156,
            0.0,
            0.0,
            2.984945030804856e-16,
            0.0,
            0.0,
            0.0
        ])

        self.sim.data.qpos[:] = self.init_qpos[:]
        self.sim.data.qvel[:] = self.init_qvel[:]
        self.sim.data.set_mocap_pos('mocap', self.sim.data.get_joint_qpos('forearm')[0:3])
        self.sim.data.set_mocap_quat('mocap', self.sim.data.get_joint_qpos('forearm')[3:7])

        self.traj_qpos = None
        self.traj_qvel = None

        # set default hand rotation
        addr = self.sim.model.get_joint_qpos_addr('forearm')
        default_mocap = self.init_qpos[addr[0]: addr[1]]

        # mocap actions 3 for position 4 for quaternion
        if not self.euler_angles:
            if not self.relative_mocap:
                mocap_low = np.array([-0.5, -1, 0, -2, -1, -1, -1])
                mocap_high = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1])
            else:
                pos_lim = 0.3   #0.05
                #pos_lim = 0.1   #0.05
                rot_lim = 0.3   #0.05
                #rot_lim = 0.1   #0.05
                mocap_low = np.array([-pos_lim, -pos_lim, -pos_lim, -1, -rot_lim, -rot_lim, -rot_lim])
                mocap_high = np.array(-mocap_low)
        else:
            if not self.relative_mocap:
                mocap_low = np.array([-0.5, -1, 0, -np.pi, -np.pi, -np.pi])
                mocap_high = np.array([0.5, 0.5, 0.5, np.pi, np.pi, np.pi])
            else:
                pos_lim = 0.08   #0.05
                #pos_lim = 0.1   #0.05
                rot_lim = np.pi / 4   #0.05
                #rot_lim = 0.1   #0.05
                mocap_low = np.array([-pos_lim, -pos_lim, -pos_lim, -rot_lim, -rot_lim, -rot_lim])
                mocap_high = np.array(-mocap_low)

        bounds = self.model.actuator_ctrlrange.copy()
        low = np.concatenate((mocap_low, bounds[:, 0]))
        high = np.concatenate((mocap_high, bounds[:, 1]))

        if self.acs_norm:
            self.acs_low = deepcopy(low)
            self.acs_high = deepcopy(high)
            low = -np.ones(low.shape)
            high = np.ones(high.shape)

        self.total_mocap_low = np.array([-1, -1, 0])
        self.total_mocap_high = np.array([1, 1, 2])

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        if self.relative_mocap:
            init_mocap = np.array([0, 0, 0, 0, 0, 0]) if self.euler_angles else np.array([0, 0, 0, 1, 0, 0, 0])
        else:
            init_mocap = np.concatenate([default_mocap[:3], rotations.quat2euler(default_mocap[3:])]) if self.euler_angles else default_mocap

        actions = np.concatenate((init_mocap, np.zeros(self.model.nu)))

        observation, _reward, done, _info = self.step(actions)
        assert not done
        self.obs_dim = observation.size

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high, dtype=np.float64)

        self.seed()

        utils.EzPickle.__init__(self)

    def _get_obs(self):
        qpos = deepcopy(self.sim.data.qpos)
        qvel = deepcopy(self.sim.data.qvel)

        names = get_sensor_hpe_names(self.sim.model)
        s_pos = get_sensor_pos(names, self.sim.data)    # 3D position of the hand sensors (21 points)

        _, names, _ = get_hand_names(self.sim.model)
        relative_pos = np.array([get_joint_qpos(self.sim, name) for name in names])
        relative_vel = np.array([get_joint_qvel(self.sim, name) for name in names])

        xpos_palm = np.array(self.get_body_com("palm").flat)
        xpos_object = np.array(self.get_body_com("Object").flat)

        quat_palm = np.array(self.sim.data.get_body_xquat("palm").flat)
        quat_object = np.array(self.sim.data.get_body_xquat("Object").flat)
        # make sure quaternions are normalised
        quat_palm = vector_norm(quat_palm)
        quat_object = vector_norm(quat_object)
        quat_diff = rotations.quat_mul(quat_palm, rotations.quat_conjugate(quat_object))
        angle_diff = rotations.quat2euler(quat_diff)  # rotation difference between hand and object expressed in euler angles

        # normalise the sensor points compared to the object
        s_pos = np.array([s_pos[i] - xpos_object for i in range(len(s_pos))]).ravel()

        palm_vel = np.array(get_joint_qvel(self.sim, 'forearm'))
        object_vel = np.array(get_joint_qvel(self.sim, 'Object'))
        vel_diff = palm_vel - object_vel


        # add contact sensors to observation
        active_dist = get_active_contacts_dist(self.sim.data, self.contact_pairs)   # current active contact
        contacts = self.get_contact_geom()                  # the contact distances of the five fingertips
        for key in active_dist:
            assert key in contacts.keys()
            if key in contacts.keys():
                contacts[key] = max(active_dist[key])


        #qpos has the absolute position of the object and the mocap
        #return np.concatenate([qpos[:].flat, qvel[:].flat, s_pos[:], xpos_palm[:], quat_palm[:]])

        # make everything relative!
        res = np.concatenate([(xpos_palm - xpos_object)[:].flat, angle_diff[:].flat, vel_diff[:].flat, relative_pos[:].flat, relative_vel[:].flat, np.fromiter(contacts.values(), dtype=float)[:].flat])

        if self.mean_obs is not None and self.std_obs is not None:
            res = (res - self. mean_obs) / self.std_obs
        return res
        #return np.concatenate([qvel[:].flat, s_pos[:], xpos_palm[2].flat, xpos_object[2].flat, rotations.quat2euler(quat_palm)[:]])
        #return np.concatenate([(xpos_palm - xpos_object)[:].flat, palm_vel[:].flat])
        #return np.concatenate([s_pos, xpos_object[:].flat, xpos_palm[:].flat, rotations.quat2euler(quat_palm)[:].flat])

    def state_vector(self):
        return self._get_obs()

    def step(self, action):

        action = np.clip(action, self.action_space.low, self.action_space.high) # action clipping, related thread https://github.com/openai/baselines/issues/121

        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        self.mocap_timestep += 1

        xpos_palm = np.array(self.get_body_com("palm").flat)
        xpos_object = np.array(self.get_body_com("Object").flat)

        d_table_object = xpos_object[2]-0.02963282 # z coordinate

        d_palm_object = np.linalg.norm(xpos_palm - xpos_object)

        # print(d_palm_object)
        # print(d_table_object)
        # if d_palm_object >= 0.12:
        #     reward = -0.1 * d_palm_object
        # else:
        reward = 5*d_table_object

        upwards = bool(xpos_palm[2] - d_table_object > 0.2)
        contact = len(get_active_contacts_dist(self.sim.data, self.contact_pairs)) > 1
        success = bool(xpos_object[2] > d_table_object          # lifting
                       and d_palm_object < 0.2                  # hand close to object
                       and xpos_object[2] > 0.2)                # object is high enough

        self._success = success or self._success
        # terminate if max timesteps reached, if hand is too far from object or if hand is lifting the object
        if self.mocap_timestep >= self.max_timestep or \
           d_palm_object > 0.25 or \
           success:
            done = True
            if self.print_msg or self.viewer is not None:
                msg = ""
                if self.mocap_timestep >= self.max_timestep:
                    msg = "Maximum steps reached "
                if d_palm_object > 0.25:
                    msg += "hand too far from object "
                if success:
                    msg += "object being lifted"
                print(msg)
        else:
            done = False

        return ob, reward, done, dict(reward_palm=d_palm_object, reward_table=d_table_object, success=success,
                                      contact=contact, upwards=upwards)

    def reset_model(self):
        prob = np.random.random() < self._epsilon

        # if init is random, it initialises randomly around the init position. Otherwise it inits from the trajectories
        random_init = not prob

        if random_init or self.traj_qpos is None or self.traj_qvel is None:
            qpos = deepcopy(self.init_qpos)  # + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
            qvel = deepcopy(self.init_qvel)
            # add randomisation
            qpos[0:3] += self.np_random.uniform(low=-self._sigma, high=self._sigma, size=3)   # hand global position
            tmp = rotations.quat2euler(qpos[3:7]) + self.np_random.uniform(low=-self._sigma / 5, high=self._sigma / 5, size=3)
            qpos[3:7] = rotations.euler2quat(tmp)                                           # hand global rotation
            qpos[7:] += self.np_random.uniform(low=-self._sigma, high=self._sigma, size=30)   # remaining joints
        else:
            assert len(self.traj_qvel) == len(self.traj_qpos)
            expert_idx = np.random.randint(0, len(self.traj_qpos))
            qpos = deepcopy(self.traj_qpos[expert_idx])
            qvel = deepcopy(self.traj_qvel[expert_idx])

        return self.reset_model_pos(qpos=qpos, qvel=qvel)

    def reset_model_pos(self, qpos, qvel):
        self.sim.data.qpos[:] = qpos[:]
        self.sim.data.qvel[:] = qvel[:]
        self.sim.data.set_mocap_pos('mocap', self.sim.data.get_joint_qpos('forearm')[0:3])
        self.sim.data.set_mocap_quat('mocap', self.sim.data.get_joint_qpos('forearm')[3:7])

        self.set_state(qpos, qvel)

        xpos_palm = np.array(self.get_body_com("palm").flat)
        xpos_object = np.array(self.get_body_com("Object").flat)

        d_palm_object = np.linalg.norm(xpos_palm - xpos_object)

        self.mocap_timestep = 0
        return self._get_obs()

    def do_simulation(self, action, n_frames):

        if self.acs_norm:
            action[:self.acs_low.size] = \
                (self.acs_high + self.acs_low) / 2 + (self.acs_high - self.acs_low) / 2 * action[:self.acs_low.size]

        mocap_quat = None
        mocap_pos = action[:self.sim.data.mocap_pos.size]

        if self.relative_mocap:
            obj_pos = self.sim.data.get_joint_qpos('forearm')

            #print('relative rot after = ' + str(action[0:7].tolist()))

            mocap_pos += obj_pos[:self.sim.data.mocap_pos.size]

            if len(action) > self.sim.data.mocap_pos.size:
                if not self.euler_angles:   # quaternions
                    action[self.sim.data.mocap_pos.size:self.sim.data.mocap_pos.size + self.sim.data.mocap_quat.size] = \
                        vector_norm(quat.q_mult(action[self.sim.data.mocap_pos.size:self.sim.data.mocap_pos.size + self.sim.data.mocap_quat.size],
                                    obj_pos[self.sim.data.mocap_pos.size:self.sim.data.mocap_pos.size + self.sim.data.mocap_quat.size]))
                else:   # euler
                    rel_q = rotations.euler2quat(action[self.sim.data.mocap_pos.size: self.sim.data.mocap_pos.size + 3])
                    #print("rot after = " + str(rel_q.tolist()))
                    total_quat = rotations.quat_mul(rel_q,
                                                    obj_pos[self.sim.data.mocap_pos.size:self.sim.data.mocap_pos.size
                                                    + self.sim.data.mocap_quat.size])
                    #print("total rot after = " + str(vector_norm(total_quat).tolist()))
                    mocap_quat = vector_norm(total_quat)
        elif self.euler_angles:
            total_quat = rotations.euler2quat(action[self.sim.data.mocap_pos.size: self.sim.data.mocap_pos.size + 3])
            mocap_quat = vector_norm(total_quat)


                #print('obj after = ' + str(obj_pos[0:4].tolist()))
            #print('rot after = ' + str(action[0:4].tolist()))

        # mocap position
        #mocap_pos = np.clip(mocap_pos, self.total_mocap_low, self.total_mocap_high)
        self.sim.data.mocap_pos[:] = mocap_pos.reshape(self.sim.data.mocap_pos.shape)

        # mocap rotation
        if len(action) > self.sim.data.mocap_pos.size:
            if mocap_quat is None:
                mocap_quat = action[self.sim.data.mocap_pos.size:self.sim.data.mocap_pos.size
                                                                 + self.sim.data.mocap_quat.size]
            ##ignore w in actions
            #mocap_quat = quat.unit_q(mocap_quat[1:])
            if not mocap_quat.any():    # check for zero quaternion
                mocap_quat = np.array([1, 0, 0, 0])
            self.sim.data.mocap_quat[:] = mocap_quat.reshape(self.sim.data.mocap_quat.shape)

        # hand relative actions
        idx = self.sim.data.mocap_quat.size if not self.euler_angles else 3
        idx += self.sim.data.mocap_pos.size
        if len(action) > idx:
            ctrl = action[idx:]
            self.sim.data.ctrl[:] = ctrl

        # render
        for _ in range(n_frames):
            try:
                self.sim.step()
            except:
                print("ERROR - NAN actions!!!!")
            if self.visual:
                self.render()

    def set_mocap_action_space(self):
        low = self.action_space.low[0:3]
        high = self.action_space.high[0:3]
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def set_expert_initialisation(self, trajectories, traj_prc=100):
        """
        Set the initial state of the environment from a set of trajectories
        :param trajectories: the trajectories to be used as initial states
        :param traj_prc: percentage of the trajectory frames that will be used as initial state
        :return:
        """
        if 'qpos' and 'qvel' in trajectories.keys():
            self.traj_qpos = []
            self.traj_qvel = []
            for i in range(len(trajectories['qpos'])):
                traj_qpos = trajectories['qpos'][i][:int(traj_prc/100 * len(trajectories['qpos'][i]))]
                traj_qvel = trajectories['qvel'][i][:int(traj_prc/100 * len(trajectories['qvel'][i]))]
                for qpos, qvel in zip(traj_qpos, traj_qvel):
                    self.traj_qpos.append(qpos)
                    self.traj_qvel.append(qvel)
        else:
            logger.warn("Expert trajectories have no qpos and qvel. Using default initial state.")

    def render(self, mode='human'):
        if not self.slowmo:
            super().render()
        else:
            from time import time
            start = time()
            while time() - start < 2:
                super().render()

    def get_contact_geom(self):
        geom_names = ['thumb3',
                      'index3',
                      'middle3',
                      'ring3',
                      'pinky3']
        res = {self.sim.model.geom_name2id(name): 0.05 for name in geom_names}
        return res

    def normalise_obs(self, data):
        if 'mean_obs' in data.keys() and len(data['mean_obs']) == self.obs_dim \
         and 'std_obs' in data.keys() and len(data['std_obs']) == self.obs_dim:
            self.mean_obs = data['mean_obs']
            self.std_obs = data['std_obs']

    def ss(self, state_dict):
        if 'qpos' in state_dict:
            self.sim.data.qpos[:] = state_dict['qpos']
        if 'qvel' in state_dict:
            self.sim.data.qvel[:] = state_dict['qvel']
        self.sim.data.set_mocap_pos('mocap', self.sim.data.get_joint_qpos('forearm')[0:3])
        self.sim.data.set_mocap_quat('mocap', self.sim.data.get_joint_qpos('forearm')[3:7])
        if 'qpos' in state_dict and 'qvel' in state_dict:
            self.set_state(state_dict['qpos'], state_dict['qvel'])

    def gs(self):
        return {'qpos': deepcopy(self.sim.data.qpos.ravel()), 'qvel': deepcopy(self.sim.data.qvel.ravel())}

    def reset(self):
        ret = super().reset()
        self._success = False
        return ret

    @property
    def success(self):
        return self._success

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value
        self.sim.nsubsteps = int(ceil(1/(self._fps * self.model.opt.timestep)))

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value