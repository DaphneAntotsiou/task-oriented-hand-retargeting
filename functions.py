__author__ = 'DafniAntotsiou'

from numpy import array
import numpy as np
import math
from similarity_transform import similarity_transform
from skinematics import quat, vector, rotmat
from copy import deepcopy
from actuator import Actuator

DEBUG_HAND = False


def get_sensor_hpe_names(m):
    _, names, _ = get_hand_names(m)
    nl = len(names)
    # clean up the names to leave only those corresponding to hpe points
    for i in range(5):
        del names[nl - (i * 4 + 4) + 1]
    del names[2]
    del names[0]

    return names


def get_hand_names(m):
    palm_name = "forearm"
    mocap_name = "mocap"
    names = [
        "wrist_PRO",        # 0
        "wrist_UDEV",       # 1
        "wrist_FLEX",       # 2
        "thumb_ABD",        # 3
        "thumb_MCP",        # 4
        "thumb_PIP",        # 5
        "thumb_DIP",        # 6
        "index_ABD",        # 7
        "index_MCP",        # 8
        "index_PIP",        # 9
        "index_DIP",        # 10
        "middle_ABD",       # 11
        "middle_MCP",       # 12
        "middle_PIP",       # 13
        "middle_DIP",       # 14
        "ring_ABD",         # 15
        "ring_MCP",         # 16
        "ring_PIP",         # 17
        "ring_DIP",         # 18
        "pinky_ABD",        # 19
        "pinky_MCP",        # 20
        "pinky_PIP",        # 21
        "pinky_DIP"         # 22
    ]

    return palm_name, names, mocap_name


# returns tuple with the actuator list and the initial hand rotation quaternion
def get_model_info(m):
    palm_name, names, mocap_name = get_hand_names(m)
    prefix_act = "A_"

    #actuator list
    idvA = [Actuator(m, prefix_act + name, 'actuator', palm_name) for name in names]
    idvA.append(Actuator(m, mocap_name, 'body', palm_name)) # add mocap

    #get initial rotation of the hand

    hand_id = m.body_name2id(palm_name)
    default_q = deepcopy(m.body_quat[hand_id])
    return idvA,  quat.Quaternion(default_q)


# native functions currently not working on windows
def get_joint_qpos(sim, name):
    addr = sim.model.get_joint_qpos_addr(name)
    if not isinstance(addr, tuple):
        return sim.data.qpos[addr]
    else:
        start_i, end_i = addr
        return sim.data.qpos[start_i:end_i]


def get_joint_qvel(sim, name):
    addr = sim.model.get_joint_qvel_addr(name)
    if not isinstance(addr, tuple):
        return sim.data.qvel[addr]
    else:
        start_i, end_i = addr
        return sim.data.qvel[start_i:end_i]


def get_sensor_pos(names, data):
    pos = deepcopy([data.get_joint_xanchor(name) for name in names])

    # approximate the DIP from the distal sites in the model
    dip_suffix = "_distal"
    finger_names = ["thumb", "index", "middle", "ring", "pinky"]
    dips = deepcopy([data.get_site_xpos(name + dip_suffix) for name in finger_names])
    for i in range(1, 6):
        pos.insert(i * 4, dips[i - 1])
    return np.asarray(pos)


def get_scale(r, b_inverse=False):
    s = np.zeros(shape=r.shape)
    for i in range(s.shape[1]):
        s[i, i] = 1 / np.linalg.norm(r[:, i]) if b_inverse else np.linalg.norm(r[:, i])

    return s


def roll_pitch_yaw(r):
    rpy = array([0, 0, 0], dtype='f8')
    sy = math.sqrt(r[2, 2]*r[2, 2] + r[1, 2] * r[1, 2])
    if sy:
        rpy[2] = -math.atan2(r[0, 1], r[0, 0])
        rpy[1] = -math.atan2(-r[0, 2], sy)
        rpy[0] = -math.atan2(r[1, 2], r[2, 2])
    else:
        # euler angle singularities
        rpy[2] = -math.atan2(-r[1, 0], r[1, 1])
        rpy[1] = -math.atan2(-r[0, 2], sy)
        rpy[0] = 0

    return rpy


def vector_norm(v):
    if v.any():
        return v / np.linalg.norm(v)
    else:
        return 0


def vector_angle(v1, v2, ref_angle=math.pi):
    v1_u = vector_norm(v1)
    v2_u = vector_norm(v2)

    c = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    angle = np.arccos(c)
    angle = angle - ref_angle if c < 0 and ref_angle == math.pi or ref_angle == 0 else ref_angle - angle
    return angle


def rot_from_vectors(v1, v2):
    v1 = vector_norm(v1)
    v2 = vector_norm(v2)
    d = np.dot(v1, v2)
    if d > 1:
        m = np.eye(3, 3)
    elif d < 1e-6 - 1:
        axis = np.cross(v1, array([0, 1, 0]))
        m = rotmat.R(axis=axis, alpha=math.pi)
    else:
        s = math.sqrt((1 + d) * 2)
        invs = 1 / s
        c = np.cross(v1, v2)
        q = array([
                s / 2
                , c[0] * invs
                , c[1] * invs
                , c[2] * invs
                ])
        m = quat.convert(q)

    return m


def fit_plane(points):
    # do least squares fit
    tmp_A = []
    tmp_b = []
    for i in range(len(points)):
        tmp_A.append([points[i][0], points[i][1], 1])
        tmp_b.append(points[i][2])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b

    return np.squeeze(np.asarray(fit))


def vec2plane_proj(v, n, d=0):
    # projection of v onto plane
    return v - (np.dot(n, v) + d) * n


# this function is missing from the current version of quat...
def rotmat2quat(r):
    return quat.Quaternion(rotmat.convert(r))


def move_mocap(point, init_pos, rot_scene=False):
    # move the hand in the scene
    c = 0.0020  # 0.0025
    new_pos = deepcopy(init_pos)
    new_pos[0] += point[0] * c
    if not rot_scene:
        new_pos[1] += point[1] * c - 0.2  # - 150 * c
        new_pos[2] += point[2] * c
    else:
        new_pos[1] = init_pos[1] + point[1] * c + 0.8  # - 150 * c
        new_pos[2] = init_pos[2] + point[2] * c - 1

    return new_pos, c


# retargeting of hpe observations (hand skeleton) to actions
def obs2actions(xyz, idvA, init_pos, default_q, default_mat, m_in, ad_hoc=True, rot_scene=False):
        if rot_scene:
            from scene_transform import hpe2mjcsrot as hpe2mjcs
        else:
            from scene_transform import hpe2mjcs

        if isinstance(xyz, list):
            xyz = np.asarray(xyz)
        if isinstance(xyz, np.ndarray) and xyz.shape == (21, 3):

            hpe_ids = array([0, 2, 6, 10, 14, 18, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21])
            hpe_ids[1:] -= 1

            skeleton = np.zeros((21, 3), dtype='float64')
            for i in range(len(hpe_ids)):
                skeleton[i] = xyz[hpe_ids[i]]

            #elif len(xyz) == 63:
            #   skeleton = np.reshape(xyz, (21, 3))

            for i in range(len(skeleton)):
                skeleton[i] = hpe2mjcs(skeleton[i])

            # move the hand in the scene
            new_pos, c = move_mocap(point=skeleton[0], init_pos=init_pos, rot_scene=rot_scene)
            new_pos = default_mat @ new_pos
            idvA[-1].set_value(new_pos)

            return calc_hand_rots(skeleton, idvA, default_q, m_in, constant=c, ad_hoc=ad_hoc)

        return None


def get_body_com(sim, body_name):
    idx = sim.model.body_names.index(body_name)
    return sim.data.com_subtree[idx]


def get_pair_contacts(model):
    geom1 = model.pair_geom1
    geom2 = model.pair_geom2
    # TODO: group the geoms into bodies
    pairs = {}
    if geom1 is not None and geom2 is not None:
        assert(len(geom1) == len(geom2))
        # group geom2 by geom1
        for elem in set(geom1):
            tmp = [geom2[i] for i in np.where(np.asarray(geom1) == elem)[0]]
            pairs[elem] = tmp
    return pairs


# get the contact distances per geometry that are currently active
def get_active_contacts_dist(data, contact_pairs):
    dist = {}
    for geom1 in contact_pairs:
        d1 = []
        for coni in range(data.ncon):
            con = data.contact[coni]
        # get distances for all active contacts with geom1
        #d1 = [data.contact[coni].dim for coni in range(data.ncon) if data.contact[coni].geom1 == geom1
        #      and data.contact[coni].geom2 == geom2 for geom2 in np.where(np.asarray(contact_pairs(geom1)))[0]]
            if (geom1 == con.geom1 and con.geom2 in contact_pairs[geom1]) \
                    or (geom1 == con.geom2 and con.geom1 in contact_pairs[geom1]):
                # contact is in the pair list
                d1.append(con.dist)

        if len(d1):
            dist[geom1] = d1
    return dist


def read_trajectories(path):
    time = []
    xyz = []
    actions = []
    obj_pos = []

    inp = open(path, "r")
    # read line into array
    for line in inp.readlines():
        line = line.replace('[', '')
        line = line.replace(']', '')
        # loop over the elements, split by whitespace
        arr = [float(i) for i in line.split()]

        if len(arr) == 101:
            time.append(arr[0])
            proxy = arr[1:64]
            xyz.append(np.reshape(proxy, (-1, 3)))
            proxy = arr[64:94]
            actions.append(proxy)
            proxy = arr[94:]
            obj_pos.append(proxy)

    return time, xyz, actions, obj_pos


def read_skeleton(skel_path):
    skel_path = skel_path.replace("\\", "/")
    arr = []
    inp = open(skel_path, "r")
    # read line into array
    for line in inp.readlines():
        line = line.replace('[', '')
        line = line.replace(']', '')
        # add a new sublist
        arr.append([])
        # loop over the elements, split by whitespace
        for i in line.split():
            # convert to integer and append to the last
            # element of the list
            arr[-1].append(float(i))
    return arr


def rotate_finger_points(skeleton, wrist_rot, idv, ad_hoc=True):
    skel_ids = []
    model_ids = []

    for k in range(5):
        finger_ids = array([0, k + 1, 6 + 3 * k, 6 + 3 * k + 1, 6 + 3 * k + 2])
        skel_ids.append(finger_ids)

        finger_ids = array([0, 3 + 4 * k, 3 + 4 * k + 1, 3 + 4 * k + 2])
        model_ids.append(finger_ids)

    # remove the wrist rotation from all the skeleton points
    inv_rot = np.linalg.inv(wrist_rot)
    for i in range(len(skeleton)):
        skeleton[i] = inv_rot @ skeleton[i]

    if DEBUG_HAND:
        print_mat(skeleton, "A1")

    # get new wrist plane
    palm_plane = fit_plane(skeleton[0:5])
    # get upward wrist plane normal
    n1 = array([-palm_plane[0], -palm_plane[1], 1])
    n1 = vector_norm(n1)
    n_rest = array([0, 0, 1])

    if DEBUG_HAND:
        print("n={};".format(n1))

    wrist_rot2 = rot_from_vectors(n_rest, n1)
    rpy = roll_pitch_yaw(wrist_rot2)

    idv[0].set_value(rpy[1])
    idv[1].set_value(rpy[2])
    idv[2].set_value(rpy[0])

    # remove secondary wrist rotation from the skeleton points
    inv_rot = np.linalg.inv(wrist_rot2)
    for i in range(len(skeleton)):
        skeleton[i] = inv_rot @ skeleton[i]

    if DEBUG_HAND:
        print_mat(skeleton, "A2")

    # assume the second rotation worked - palm plane is now perpendicular to [0,0,1]
    n1 = n_rest

    for k in range(len(skel_ids)):
        # 3D angle of the last two points of each finger is their total actuator angle

        # ***** DIP point *****
        v1 = skeleton[skel_ids[k][4]] - skeleton[skel_ids[k][3]]
        v2 = skeleton[skel_ids[k][2]] - skeleton[skel_ids[k][3]]

        angle = vector_angle(v1, v2)
        if not k:
            nv = np.cross(v2, v1)
            angle = -angle if nv[2] * angle < 0 else angle
        else:
            angle = math.fabs(angle)

        idv[6 + k * 4].set_value(angle)
        ########
        if not k:
            idv[5 + k * 4].set_value(angle)
        ######

        # pip_limit = angle if k else angle / 2

        # ***** PIP point *****
        v1 = skeleton[skel_ids[k][3]] - skeleton[skel_ids[k][2]]
        v2 = skeleton[skel_ids[k][1]] - skeleton[skel_ids[k][2]]

        angle = math.fabs(vector_angle(v1, v2))
        if k:
            idv[5 + k * 4].set_value(angle)
            if ad_hoc:
                # hack DIP point of fingers because hpe does not register them well
                tmp = max(angle / 2, idv[6 + k * 4].get_value())
                idv[6 + k * 4].set_value(tmp)

        # ***** MCP & ABD points *****
        # plane normal is perpendicular to palm plane so also perpendicular to the resting position of MCP
        idx = 1 if k else 2
        o = vec2plane_proj(skeleton[skel_ids[k][1]], n1, palm_plane[2])  # projection of MCP onto palm plane - this will be the new origin for this angle

        if DEBUG_HAND:
            print("v1p=[{};{}];".format(o, v1_proj))
            print("v1=[{};{}];".format(o, skeleton[skel_ids[k][2]]))

        v1 = skeleton[skel_ids[k][idx+1]] - o                      # centre on new origin

        if k:
            # all the fingers except thumb
            v1_proj = vec2plane_proj(skeleton[skel_ids[k][idx + 1]], n1,
                                     palm_plane[2])  # projection of v1 onto the plane
            v1_proj -= o  # centre on new origin

            fd = array([0, -1, 0])   # default direction of the fingers

            # ** ABD ** #
            v2_proj = vec2plane_proj(skeleton[skel_ids[k][idx + 2]], n1,
                                     palm_plane[2])  # projection of v1 onto the plane
            v2_proj -= o
            angle = vector_angle(v2_proj, fd, 0)
            angle = -angle if v2_proj[0] > fd[0] else angle
            idv[3 + k * 4].set_value(angle)

            # ** MCP ** #
            angle = vector_angle(v1, v1_proj, 0)
            angle = -angle if v1[2] > v1_proj[2] else angle
            idv[4 + k * 4].set_value(angle)

        else:
            # thumb
            fd = array([1, 0, 0])  # default direction of the fingers

            n2 = array([0, -1, 0])  # perpendicular to palm plane
            v1p2 = vec2plane_proj(v1, n2)  # projection of v1 onto plane with normal n2

            # ** ABD ** #
            angle = vector_angle(v1p2, fd, 0)
            angle = -angle if v1p2[2] > fd[2] else angle
            idv[3 + k * 4].set_value(angle)

            # ** MCP ** #
            angle = vector_angle(v1p2, v1, 0)
            angle = -angle if v1[1] > v1p2[1] else angle
            idv[4 + k * 4].set_value(angle)

    return idv


def print_mat(m, name=None):
    if name:
        print('{} ='.format(name), end='')
    print(np.matrix(m), end='')
    print(";\n")


def calc_hand_rots(skeleton, idv, default_q, m_in, constant=None, ad_hoc=True):
    # normalise the skeleton for local angles with wrist as the origin
    wrist_pos = deepcopy(skeleton[0][:])  # normalise with wrist as (0, 0, 0)
    for i in range(len(skeleton)):
        skeleton[i][:] -= wrist_pos[:]

    if DEBUG_HAND:
        print_mat(skeleton, "B")

    # get palm rotation
    m_out = np.zeros(shape=np.shape(m_in))

    for j in range(2, 6):
        m_out[j - 1, 0] = skeleton[j][0]
        m_out[j - 1, 1] = skeleton[j][1]
        m_out[j - 1, 2] = skeleton[j][2]

    wrist_rot, translation = similarity_transform(m_in, m_out)
    scale = get_scale(wrist_rot, True)

    wrist_rot = np.matrix(wrist_rot)
    #wrist_rot = scale * wrist_rot
    #q = quat.Quaternion(scale @ wrist_rot, inType='rotmat')
    q = rotmat2quat(scale @ wrist_rot)
    # q = Quaternion(matrix=wrist_rot)
    q = default_q * q
    #q.quat.normalize

    idv[-1].set_value(np.squeeze(q.values), is_quat=True)

    if not constant:
        translation = np.matmul(scale, translation)
    else:
        translation *= constant

    translation = vector.rotate_vector(translation, np.squeeze(default_q.values))

    idv[-1].set_value(translation + idv[-1].get_value())

    # *** find finger point rotation *****
    idv = rotate_finger_points(skeleton, wrist_rot, idv, ad_hoc=ad_hoc)

    if DEBUG_HAND:
        print("scale = {};".format(scale))

    return idv


def get_joint_state(name, data):
    if name is not None and data is not None:
        try:
            # object exists
            obj_pos = deepcopy(data.get_joint_qpos(name))
            obj_vel = deepcopy(data.get_joint_qvel(name))
            return obj_pos, obj_vel
        except ValueError:
            pass
    return None


def read_npz(path):
    data = np.load(path)
    res = dict(data)
    data.close()
    return res
