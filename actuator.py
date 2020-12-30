__author__ = 'DafniAntotsiou'

import mujoco_py
from numpy import array
from copy import deepcopy
import numpy as np


def GetBodyPosDist(bodyId, refbodyId, m):
    bodyPos = array([0, 0, 0], dtype='f8')

    if refbodyId < 0:
        return bodyPos

    while bodyId > refbodyId:
        bodyPos += m.body_pos[bodyId]
        bodyId = m.body_parentid[bodyId]

    return bodyPos


def GetJointBodyDist(jointId, bodyId, m):
    jointPos = array([0, 0, 0], dtype='f8')

    if jointId < 0 or bodyId < 0:
        return jointPos

    jointPos += m.jnt_pos[jointId]
    jointPos += GetBodyPosDist(m.jnt_bodyid[jointId], bodyId, m)

    return jointPos


#Quaternion = [w,x,y,z]
def GetBodyRotDist(bodyId, refbodyId, m):

    bodyRot = array([1, 0, 0, 0], dtype='f8')

    if refbodyId < 0:
        return bodyRot

    while bodyId > refbodyId:

        currRot = m.body_quat[bodyId]

        tmp = deepcopy(bodyRot)

        bodyRot[0] = currRot[0] * tmp[0] - currRot[1] * tmp[1] - currRot[2] * tmp[2] - currRot[3] * tmp[3]
        bodyRot[1] = currRot[0] * tmp[1] - currRot[1] * tmp[0] - currRot[2] * tmp[3] - currRot[3] * tmp[2]
        bodyRot[2] = currRot[0] * tmp[2] - currRot[1] * tmp[2] - currRot[2] * tmp[0] - currRot[3] * tmp[1]
        bodyRot[3] = currRot[0] * tmp[3] - currRot[1] * tmp[3] - currRot[2] * tmp[1] - currRot[3] * tmp[0]

        bodyId = m.body_parentid[bodyId]

    bodyRot = bodyRot.conjugate()

    return bodyRot


def GetJointBodyRot(jointId, bodyId, m):
    jointRot = array([1, 0, 0, 0], dtype='f8')
    if jointId < 0 or bodyId < 0:
        return jointRot

    jointRot = GetBodyRotDist(m.jnt_bodyid[jointId], bodyId, m)

    return jointRot


# class that handles mujoco actuators, joints and mocaps
class Actuator(object):

    def __init__(self, m=None, name=None, mjtype=None, palm_name=None):
        self.type = mjtype
        self.default_pos = array([0, 0, 0], dtype='f8')
        self.default_quat = array([1, 0, 0, 0], dtype='f8')
        self.quat = None
        self.id = None
        self.bodyid = None
        self.value = None
        self.name = name
        self.min = -1000
        self.max = 1000

        if m and name and mjtype and palm_name:
            main_id = m.body_name2id(palm_name)
            if mjtype == 'body':
                self.id = m.body_name2id(name)
                self.bodyid = self.id
                if name.find("mocap") >= 0:
                    self.id = m.body_mocapid[self.id]
                    self.quat = array([1, 0, 0, 0])
                self.value = array([0, 0, 0], dtype='f8')
                self.default_pos = GetBodyPosDist(self.id, main_id, m)
                self.default_quat = GetBodyRotDist(self.id, main_id, m)
            elif mjtype == 'joint':
                self.id = m.joint_name2id(name)
                self.value = np.float64(0)
                self.default_pos = GetJointBodyDist(self.id, main_id, m)
                self.default_quat = GetJointBodyRot(self.id, main_id, m)

                self.min = self.m.jnt_range[self.id][0]
                self.max = self.m.jnt_range[self.id][1]
                self.bodyid = m.jnt_bodyid[self.id]

            elif mjtype == 'actuator':
                self.id = m.actuator_name2id(name)
                self.value = np.float64(0)

                # assume the actuator name is A_ + joint name
                idx = name.find("A_")
                if idx >= 0:
                    j_name = name[idx + 2:]
                    joint_id = m.joint_name2id(j_name)
                    self.bodyid = m.jnt_bodyid[joint_id]
                    self.default_pos = GetJointBodyDist(joint_id, main_id, m)
                    self.default_quat = GetJointBodyRot(joint_id, main_id, m)

                self.min = m.actuator_ctrlrange[self.id][0]
                self.max = m.actuator_ctrlrange[self.id][1]

    def set_value(self, val, safe=False, is_quat=False):
        if self.type == 'body':
            if not is_quat:
                self.value = deepcopy(val)
            else:
                self.quat = deepcopy(val)
        elif safe and (self.type == 'actuator' or self.type == 'joint'):
            self.value = max(val, self.min)
            self.value = min(self.value, self.max)
        else:
            self.value = deepcopy(val)

    def get_limits(self):
        return self.min, self.max

    def get_value(self, is_quat=False):
        if not is_quat:
            return deepcopy(self.value)
        else:
            return deepcopy(self.quat)

    def assign(self, sim):
        self.set_value(self.value, True)

        if self.type == 'joint':
            sim.data.qpos[sim.model.jnt_qposadr[self.id]] = deepcopy(self.value)
        elif self.type == 'actuator':
            sim.data.ctrl[self.id] = deepcopy(self.value)
        elif self.type == 'body':
            sim.data.mocap_pos[self.id] = deepcopy(self.value)
            sim.data.mocap_quat[self.id] = self.quat
        return sim

    def get_id(self):
        return self.id

    def get_pos(self):
        return deepcopy(self.default_pos)

    def get_quat(self):
        return deepcopy(self.default_quat)

    # deep copy of the value and quat attributes
    def copy(self):
        res = Actuator()
        res.value = np.copy(self.value)
        res.quat = np.copy(self.quat)
        res.type = self.type
        res.default_pos = self.default_pos
        res.default_quat = self.default_quat
        res.id = self.id

        return res

    def get_val_from_sim(self, sim):

        if self.type == 'joint':
            self.value = deepcopy(sim.data.qpos[sim.model.jnt_qposadr[self.id]])
        elif self.type == 'actuator':
            self.value = deepcopy(sim.data.ctrl[self.id])
        elif self.type == 'body':
            self.value = deepcopy(sim.data.mocap_pos[self.id])
            self.quat = deepcopy(sim.data.mocap_quat[self.id])
