__author__ = 'DafniAntotsiou'

''' This PSO implementation is heavily influenced by tisimst's pso implementation 
(https://github.com/tisimst/pyswarm/blob/master/pyswarm/pso.py) '''

from functions import get_sensor_hpe_names, get_sensor_pos, read_trajectories, \
    get_model_info, read_skeleton, rotmat2quat, obs2actions, get_pair_contacts, \
    get_active_contacts_dist, vector_angle, move_mocap, read_npz

import numpy as np
from copy import deepcopy
import mujoco_py as mp
from mujoco_py.builder import functions
from mjviewerext import MjViewerExt
from time import time
import os


def obj_func(x, y, pos_range=None, contact_dist=None, contact_pairs=None, c_task=None, c_angle=None):
    """
    Objective function

    :param x: position of points in the model
    :param y: position of target points
    :param pos_range: maximum distance detected
    :param contact_dist: current contact distances between geoms and target
    :param contact_pairs: contact geoms-target pairs in the model
    :param c_task: custom contact coefficient. Is set automatically if None
    :param c_angle: custom angle coefficient. Is set automatically if None
    :return: the objective function of the particle
    """
    if c_task is None or c_angle is None:
        if not contact_pairs:
            # no pairs in the xml model
            c_c = 0
            c_a = 0.25
            c_p = 0.75
        elif not contact_dist:
            # fingers are not close to any object
            c_c = 0
            c_a = 0.25
            c_p = 0.75
        else:
            # fingers are close to an object
            c_c = 0.8
            c_a = 0.1
            c_p = 0.1
    else:
        c_c = c_task
        c_a = c_angle * (1 - c_task)
        c_p = (1 - c_angle) * (1 - c_task)

    margin = 0.04   # maximum distance detected for contact
    c = 0.004

    E = c_a * E_a(x, y, e=False)\
        + c_p * E_p(x, y, pos_range=pos_range, tips_only=False, e=False, tips_weight=3, thumb_weight=10) \
        + c_c * E_c(contact_dist=contact_dist, margin=margin, constant=c, missing_weight=2)
    return E


def E_c(contact_dist, margin, constant, missing_weight=1):
    """
    Contact energy function

    :param contact_dist: dictionary of geom id and its distance from the target
    :param margin:  maximum distance between geom and target detected
    :param constant: constant dist we ideally want the geoms to "invade" target
    :param missing_weight: weight of geoms that are too far away from target
    :return: normalised contact energy function
    """
    # add palm
    palm_w = 3  # palm weight to make it more important than the rest of the tips (must be integer)
    if contact_dist is not None and 12 in contact_dist:  # palm id
        for i in range(palm_w - 1):
            contact_dist[12 + (i+1) * 100] = contact_dist[12]   # add identical palm entries for the mean
    total = 5 + palm_w
    if contact_dist is not None:
        s = (total - len(contact_dist)) * missing_weight * ((margin + constant)**2)  # punish for those that are not even in range
        for key in contact_dist:
            # ideally the distance is less than zero, so add constant to make it positive
            s += (max(max(contact_dist[key]) + constant, 0)) ** 2  # we want it to be less than 0 so there applied force
        # normalise
        # coeff = (len(contact_dist) + (5 - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
        s /= (len(contact_dist) + (total - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
        return s
    else:
        return 1


def E_p(x, y, pos_range=None, tips_only=False, e=False, tips_weight=1, thumb_weight=None):
    """
    Position energy function

    :param x: position of points in the model
    :param y: position of target points
    :param pos_range: maximum distance detected
    :param tips_only: minimise distance between all the hand points or the tips only
    :param e: is exponential mse or linear
    :param tips_weight: weight of the tips
    :param thumb_weight: weight of the thumb. Is tips_weight if None
    :return: the normalised position energy function
    """
    c = 10
    tips_ids = [i * 4 for i in range(1, 6)]  # fingertips
    if tips_only:
        ids = tips_ids
        mse = (np.linalg.norm(x[ids, :] - y[ids, :], axis=1) ** 2).mean()
    else:
        thumb_weight = tips_weight if thumb_weight is None else thumb_weight
        ids = range(len(x))
        diff = (np.linalg.norm(x[ids, :]-y[ids, :], axis=1)**2)
        diff[tips_ids[1:]] *= tips_weight
        diff[tips_ids[0]] *= thumb_weight
        s = sum(diff)
        total = len(x) - len(tips_ids) + (len(tips_ids) - 1) * tips_weight + thumb_weight
        mse = s / total
    # mse = ((x - y) ** 2).mean(axis=None)
    if pos_range is not None:
        if e:
            mse = np.exp(pos_range * c * mse) - 1
        else:
            # normalise the distances by dividing with the range
            mse /= (pos_range**2)
    return mse


def E_a(x, y, e=False):
    """
    Angle energy function

    :param x: position of points in the model
    :param y: position of target points
    :param e: is exponential mse or linear
    :return: the normalised angle energy function
    """

    # get global angles between the joint points
    a1 = []
    a2 = []
    c = 10
    for i in range(5):
        a1.append(get_joint_angle(x[0], x[i * 4 + 2], x[i * 4 + 1]))
        a2.append(get_joint_angle(y[0], y[i * 4 + 2], y[i * 4 + 1]))
        for j in range(2):
            a1.append(get_joint_angle(x[i * 4 + j + 1], x[i * 4 + j + 3], x[i * 4 + 2 + j]))
            a2.append(get_joint_angle(y[i * 4 + j + 1], y[i * 4 + j + 3], y[i * 4 + 2 + j]))

    mse = ((np.asarray(a1) - np.asarray(a2)) ** 2).mean(axis=None)
    if e:
        mse = np.exp(np.pi * c * mse) - 1
    else:
        # maximum difference between angles is pi, so normalise mse by dividing with pi
        mse /= (np.pi**2)
    return mse


def get_joint_angle(v1, v2, o):
    # get the vector angle between v1 and v2 centred at origin o
    return vector_angle(v1 - o, v2 - o, ref_angle=0)


def fit(actuators, vals, obs, sim, sensor_names, norm=True, fps=60, viewer=None,
        contact_pairs=None, pos_range=None, initial_act=None, initial_state=None, objects=None, c_task=None, c_angle=None):
    """
        fit the particle's position according to its objective function

    :param actuators: actuators list of hand model
    :param vals: position of current particle
    :param obs: position of target points
    :param sim: mujoco simulation
    :param sensor_names: names of model geom corresponding to obs
    :param norm: normalise by wrist position
    :param fps: simulation fps
    :param viewer: visualise simulation if set
    :param contact_pairs: the contact pair geoms of the hand model
    :param pos_range: maximum distance detected for position energy function
    :param initial_act: initial actuator, the state of which will be the initial state of the simulation
    :param initial_state: initial state of simulation
    :param objects: the target object geom and its state
    :param c_task: custom contact coefficient. Is set automatically if None
    :param c_angle: custom angle coefficient. Is set automatically if None
    :return: objective function of current particle
    """
    p_res = get_particle_obs(vals, actuators, sim, sensor_names, fps=fps, viewer=viewer, initial_act=initial_act,
                             initial_state=initial_state, objects=objects)

    if norm:
        # normalise positions compared to wrist
        init_pos = deepcopy(p_res[0])
        for i in range(len(p_res)):
            p_res[i] -= init_pos

    # get current contact info for intention energy function
    if contact_pairs is not None:
        dist = get_active_contacts_dist(data=sim.data, contact_pairs=contact_pairs)
    else:
        dist = None

    return obj_func(p_res, obs, pos_range=pos_range, contact_dist=dist, contact_pairs=contact_pairs,
                    c_task=c_task, c_angle=c_angle)


def pso(params, obs, model, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, norm=True, fps=10, visualise=False, default_mat=None, contact=False,
        hybrid_prc=None, initial_act=None, hybrid_space=False, objects=None, initial_state=None,
        save_pose=False, rot_scene=False, c_task=None, c_angle=None):
    """
    Perform a particle swarm optimization (PSO)

    Parameters
    ==========

    params : list
        actuators of model in the simulation
    obs : array
        skeleton 3D points of observation (i.e. from a hpe)
    model : mujoco model
        hand model file for the mujoco simulation
    Optional
    ========
    ieqcons : list
        A list of functions of length n such that ieqcons[j](x,*args) >= 0.0 in
        a successfully optimized problem (Default: [])
    f_ieqcons : function
        Returns a 1-D array in which each element must be greater or equal
        to 0.0 in a successfully optimized problem. If f_ieqcons is specified,
        ieqcons is ignored (Default: None)
    args : tuple
        Additional arguments passed to objective and constraint functions
        (Default: empty tuple)
    kwargs : dict
        Additional keyword arguments passed to objective and constraint
        functions (Default: empty dict)
    swarmsize : int
        The number of particles in the swarm (Default: 100)
    omega : scalar
        Particle velocity scaling factor (Default: 0.5)
    phip : scalar
        Scaling factor to search away from the particle's best known position
        (Default: 0.5)
    phig : scalar
        Scaling factor to search away from the swarm's best known position
        (Default: 0.5)
    maxiter : int
        The maximum number of iterations for the swarm to search (Default: 100)
    minstep : scalar
        The minimum stepsize of swarm's best position before the search
        terminates (Default: 1e-8)
    minfunc : scalar
        The minimum change of swarm's best objective value before the search
        terminates (Default: 1e-8)
    debug : boolean
        If True, progress statements will be displayed every iteration
        (Default: False)
    default_mat : matrix
        If set, the pso will remove this rotation from the observations.
        That way, the actions returned should then be rotated back
    initial_act : actuators
        If set and initial_state is not set, it sets the initial state of each iteration
        based on these actuators
    initial_state: [qpos,qvel]
        If set, it resets each iteration to this particular state
    hybrid_space : boolean
        If set, it enables search around the IK pose
    hybrid_prc : int
        The percentage of search space around the IK pose, when bybrid_space is set
    contact : bool
        If the contact energy function will be part of the objective function
    visualise : bool
        visualise the pso search
    norm : bool
        normalise by wrist position
    objects : dictionary
        the target object geom and its state

    Returns
    =======
    g : array
        The swarm's best known position (optimal design)
    f : scalar
        The objective value at ``g``

    """
    if rot_scene:
        from scene_transform import hpe2mjcsrot as hpe2mjcs
    else:
        from scene_transform import hpe2mjcs
    #start = time()
    # get bounds from the actuators
    lb = []
    ub = []
    for i in range(len(params)):
        if params[i].type.lower == 'body'.lower:
            min_val = -1
            max_val = 1
            # is quaternion
            if hybrid_space:
                val = params[i].get_value(is_quat=True)
                for elem in val:
                    #lb.append(max(min_val, elem - abs(hybrid_prc / 100.0 * (max_val - min_val))))
                    lb.append(max(min_val, elem))   # only positive action changes
                    ub.append(min(max_val, elem + abs(hybrid_prc / 100.0 * 2 * (max_val - min_val))))
            else:
                for j in range(4):
                    lb.append(min_val)
                    ub.append(max_val)
        else:
            min_val, max_val = params[i].get_limits()
            if hybrid_space:
                val = params[i].get_value()
                lb.append(max(min_val, val - abs(hybrid_prc / 100.0 * 2.0 * (max_val - min_val))))
                ub.append(min(max_val, val + abs(hybrid_prc / 100.0 * 2.0 * (max_val - min_val))))
            else:
                lb.append(min_val)
                ub.append(max_val)



    assert len(lb) == len(ub), 'Lower- and upper-bounds must be the same length'
    # assert hasattr(func, '__call__'), 'Invalid function handle'
    lb = np.array(lb)
    ub = np.array(ub)
    assert np.all(ub >= lb), 'All upper-bound values must be greater than lower-bound values'

    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Check for constraint function(s) #########################################
    # obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            if debug:
                print('No constraints given.')
            cons = lambda x: np.array([0])
        else:
            if debug:
                print('Converting ieqcons to a single constraint function')
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        if debug:
            print('Single constraint function given in f_ieqcons')
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))

    def is_feasible(x):
        check = np.all(cons(x) >= 0)
        return check

    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    x = np.random.rand(S, D)  # particle positions

    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value

    #### add simulation variables
    # start simulation environment
    #nsubstep = int(ceil(1 / (fps * model.opt.timestep)))
    #sim = mp.MjSim(model, nsubsteps=nsubstep)
    sim = mp.MjSim(model)                   # environment
    snames = get_sensor_hpe_names(model)    # sensor names
    sim.step()  # initialise all the values

    if contact:
        pairs = get_pair_contacts(model=model)  # contact pairs
    else:
        pairs = None

    if visualise:
        viewer = MjViewerExt(sim)
    else:
        viewer = None

    if initial_state is None and initial_act is not None:
        # get the initial state of the simulation to reset for every particle
        initial_state = get_ref_state(initial_act, sim, objects=objects)

    # transform the obs in simulation space
    xyz = []
    for elem in obs:
        xyz.append(hpe2mjcs(elem))

    xyz = np.asarray(xyz)

    if norm:    # normalise by wrist position
        init_pos = deepcopy(xyz[0])
        for i in range(len(xyz)):
            xyz[i] -= init_pos


    # get default position of actuators to get the scale between the two sets
    s_pos = get_sensor_pos(snames, sim.data)

    if norm:
        # normalise positions compared to wrist
        init_pos = deepcopy(s_pos[0])
        for i in range(len(s_pos)):
            s_pos[i] -= init_pos

    # scale the observation in the sensor space
    scale, max_pos, max_xyz = get_2hand_scale(s_pos, xyz)
    xyz *= scale
    max_xyz *= scale
    # rotate the observations by default_mat if specified
    if default_mat is not None:
        for i in range(len(xyz)):
            xyz[i] = default_mat @ xyz[i]

    if save_pose:
        if not os.path.exists("pose"):
            os.makedirs("pose")
        save_skeleton(xyz, "pose/hpe_pso_skel.txt")

    ######
    #stop = time() - start
    #print("initialisation time: " + str(stop))

    for i in range(S):
        # Initialize the particle's position
        if hybrid_prc is None:
            x[i, :] = lb + x[i, :] * (ub - lb)
        else:
            for j in range(len(x[i, :])):
                x[i, j] = max(min(x[i, j], ub[j]), lb[j])

        # Initialize the particle's best known position
        p[i, :] = x[i, :]

        # Calculate the objective's value at the current particle's position
        #start = time()

        fp[i] = fit(actuators=params, obs=xyz, sim=sim, sensor_names=snames, vals=p[i, :], norm=norm, fps=fps,
                    viewer=viewer, pos_range=max_pos + max_xyz, contact_pairs=pairs, initial_act=initial_act,
                    initial_state=initial_state, objects=objects, c_task=c_task, c_angle=c_angle)

        #stop = time() - start
        #print("fit time: " + str(stop))
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i == 0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i] < fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()

        # Initialize the particle's velocity
        v[i, :] = vlow + np.random.rand(D) * (vhigh - vlow)

    # Iterate until termination criterion met ##################################
    it = 1
    while it <= maxiter:
        rp = np.random.uniform(size=(S, D))
        rg = np.random.uniform(size=(S, D))
        for i in range(S):

            # Update the particle's velocity
            v[i, :] = omega * v[i, :] + phip * rp[i, :] * (p[i, :] - x[i, :]) + \
                      phig * rg[i, :] * (g - x[i, :])

            # Update the particle's position, correcting lower and upper bound
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :] < lb
            mark2 = x[i, :] > ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
            # get environment results of particle
            #start = time()
            fx = fit(actuators=params, obs=xyz, sim=sim, sensor_names=snames, vals=x[i, :], norm=norm, fps=fps,
                     viewer=viewer, pos_range=max_pos + max_xyz, contact_pairs=pairs, initial_act=initial_act,
                     initial_state=initial_state, objects=objects, c_task=c_task, c_angle=c_angle)

            #stop = time() - start
            #print("fit time: " + str(stop))
            # Compare particle's best position (if constraints are satisfied)
            if fx < fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx < fg:
                    if debug:
                        print('New best for swarm at iteration {:}: {:} {:}'.format(it, x[i, :], fx))

                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g - tmp) ** 2))
                    if np.abs(fg - fx) <= minfunc:
                        if debug:
                            print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize <= minstep:
                        if debug:
                            print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx

        if debug:
            print('Best after iteration {:}: {:} {:}'.format(it, g, fg))
        it += 1

    if debug:
        print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))

    if not is_feasible(g):
        print("However, the optimization couldn't find a feasible design. Sorry")
    return g, fg


def particle2actuator(p, act):
    """
    particle position to model actuators
    :param p: particle
    :param act: list of model actuators
    :return: actuators of particle
    """
    if len(p) >= len(act):
        i = 0
        for z in range(len(act)):
            if act[z].type.lower == 'body'.lower and i + 4 <= len(p):
                # process rotation actuator
                quat = np.array([p[i], p[i + 1], p[i + 2], p[i + 3]])
                if quat.any():
                    quat /= np.linalg.norm(quat)  # normalise quaternion
                act[z].set_value(quat, is_quat=True)
                i += 4
            else:
                act[z].set_value(p[i])
                i += 1
    return act


def get_ref_state(actuators, sim, objects=None):
    for act in actuators:
        act.assign(sim)

    sim_start = sim.data.time
    while sim.data.time - sim_start < 1.0 / 10:
        if objects is not None:
            for obj_name in objects:
                if len(objects[obj_name]) == 2:
                    try:
                        # set object position
                        sim.data.set_joint_qpos(obj_name, objects[obj_name][0])
                        sim.data.set_joint_qvel(obj_name, objects[obj_name][1])
                    except ValueError:
                        pass
        sim.step()

    # get the state of the simulator
    qpos = deepcopy(sim.data.qpos)
    qvel = deepcopy(sim.data.qvel)
    #act = deepcopy(sim.data.act)

    return qpos, qvel#, act


def get_particle_obs(p, act, sim, obs_names, fps=None, viewer=None, norm=True, initial_act=None,
                     initial_state=None, objects=None):
    """
    get the skeleton points (observations) of the current particle in the simulation
    :param p: particle point
    :param act: model actuators
    :param sim: mujoco simulation
    :param obs_names:
    :param fps: simulation fps
    :param viewer: mujoco viewer of current particle position
    :param norm:
    :param initial_act: set initial state of simulation to this initial state of the actuators
    :param initial_state: initial state of simulation
    :param objects:
    :return: skeleton points of current particle
    """
    # reset the environment
    if initial_state is None:
        functions.mj_resetData(sim.model, sim.data)
    else:
        # reset to initial state
        functions.mju_copy(sim.data.qpos, initial_state[0], sim.model.nq)
        functions.mju_copy(sim.data.qvel, initial_state[1], sim.model.nv)

    if p is not None:
        # apply the values of particle p through the actuators act to environment sim
        act = particle2actuator(p, act)

    if initial_act is not None:
        for i in range(len(initial_act)):
            sim = initial_act[i].assign(sim)

    for i in range(len(act)):
        sim = act[i].assign(sim)

    sim_start = sim.data.time

    if fps is not None:
        while sim.data.time - sim_start < 1.0 / fps:
            sim.step()

    if viewer:
        viewer.render()

    # get position of sensors
    s_pos = get_sensor_pos(obs_names, sim.data)
    if norm:    # normalise by the wrist position
        init_pos = deepcopy(s_pos[0])
        s_pos = np.array([s_pos[i] - init_pos for i in range(len(s_pos))])
    return s_pos


# get scale for y to match x
def get_2set_scale(x, y):
    if len(x) == len(y):
        # get distances
        d1 = np.array([np.linalg.norm(v) for v in x])
        d2 = np.array([np.linalg.norm(v) for v in y])

        zero_ids = d1 == 0
        d1[zero_ids] = 1e-8

        zero_ids = d2 == 0
        d2[zero_ids] = 1e-8

        div = d1/d2
        return np.mean(div)


# get scale for y to match joint length of x
def get_2hand_scale(x, y):
    if len(x) == len(y):
        # d1 = np.array([np.linalg.norm(x[i + 1] - x[i]) for i in range(len(x) - 1)])
        # d2 = np.array([np.linalg.norm(y[i + 1] - y[i]) for i in range(len(y) - 1)])
        d1 = []
        d2 = []
        maxd1 = 0
        maxd2 = 0
        for i in range(5):
            # separate fingers
            idx = i * 4 + 1
            d1.append(np.linalg.norm(x[idx] - x[0]))
            d2.append(np.linalg.norm(y[idx] - y[0]))
            for j in range(3):
                d1.append(np.linalg.norm(x[idx + j + 1] - x[idx + j]))
                d2.append(np.linalg.norm(y[idx + j + 1] - y[idx + j]))

            # get maximum distance of finger from wrist
            finger_max1 = sum(d1[-4:-1])
            finger_max2 = sum(d2[-4:-1])
            maxd1 = max(maxd1, finger_max1)
            maxd2 = max(maxd2, finger_max2)
        return get_2set_scale(d1, d2), maxd1, maxd2


def save_skeleton(skeleton, file_path):
    np.savetxt(file_path, skeleton, delimiter=',')


def visualise(actions, actuators, model, save_pose=False, initial_act=None, initial_state=None):
    sim = mp.MjSim(model)  # environment
    viewer = MjViewerExt(sim)

    snames = get_sensor_hpe_names(model)  # sensor names
    p_res = get_particle_obs(p=actions, act=actuators, sim=sim, obs_names=snames, fps=10, viewer=viewer,
                             initial_act=initial_act, initial_state=initial_state)

    if save_pose:
        # save pso pose

        # normalise positions compared to wrist
        init_pos = deepcopy(p_res[0])
        for i in range(len(p_res)):
            p_res[i] -= init_pos

        if not os.path.exists("pose"):
            os.makedirs("pose")
        save_skeleton(p_res, "pose/pso_skel.txt")

    obj_name = "Object"

    try:
        obj_id = sim.model.joint_name2id(obj_name)
    except ValueError:
        obj_id = -1
    if obj_id >= 0:
        # object exists
        sim.step()
        init_pos = deepcopy(sim.data.get_joint_qpos(obj_name))
        init_vel = deepcopy(sim.data.get_joint_qvel(obj_name))

    sim_start = sim.data.time
    while True:
        sim.step()
        viewer.render()
