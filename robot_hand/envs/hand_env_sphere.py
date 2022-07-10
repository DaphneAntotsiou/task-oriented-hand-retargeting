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
from robot_hand.envs.hand_env import HandEnv


class HandEnvSphere(HandEnv):

    def __init__(self):

        model_path = "../../model/MPL/MPL_Sphere_6.xml"
        full_path = model_path
        # custom init
        # if model_path.startswith("/"):
        #     fullpath = model_path
        # else:
        #     fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        super().__init__(fullpath)
