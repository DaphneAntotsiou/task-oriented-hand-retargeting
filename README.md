# Task Oriented Hand Motion Retargeting for Dexterous Manipulation Imitation
Python 3.x implementation of the retargeting system presented in the paper "[Task Oriented Hand Motion Retargeting for Dexterous Manipulation Imitation](https://arxiv.org/abs/1810.01845)".

Given a set of HPE trajectories, the system applies PSO task oriented optimisation and produces corrected ones.

The HPE trajectories were recorded using the 21-point HPE presented in the paper "[Spatial Attention Deep Net with Partial PSO for Hierarchical Hybrid Hand Pose Estimation](https://arxiv.org/abs/1604.03334)".  

## Requirements
The system requires the following packages:
* mujoco-py 1.50
* scipy
* scikit-kinematics
* filterpy
* numpy

## Instructions

To apply the task oriented optimisation on a set of trajectories in the directory "trajectories" run the following:
```
python3 optimise_trajectories.py --traj_path trajectories
```

To replay the trajectories in the "trajectories" folder (and optionally the optimised trajectories in the "trajectories/result" folder side by side) run the following:
```
python3 replay_trajectories.py --traj_path trajectories --opt_dir trajectories/result
```    

## Sphere Grasping Dataset

A dataset of 161 grasping demonstrations was acquired using the proposed method. The dataset can be found in the "dataset" folder.
The dataset.npz file contains a dictionary with the following 5 keys:
* obs: the observations (states) of the trajectories. The state space has been normalised by state_norm.npz.
* acs: the actions of the trajectories
* qpos: part of the detailed mujoco state of the environment for demonstration reproducibility
* qvel: part of the detailed mujoco state of the environment for demonstration reproducibility
* ep_rets: the RL rewards of the trajectories

### State space

The state space consists of proprioceptive features, such as current pose and velocity of virtual model joints, object features such as position and pose relative to hand model, and contact features. In other words:

![equation](https://latex.codecogs.com/svg.image?%5Cbegin%7Bequation%7D%20s_t%20=%20%5Bhand_%7Bjoints%7D;%20hand_%7Bvelocities%7D;%20hand_%7Bpos%7D%20-%20object_%7Bpos%7D;%20hand_%7Bvelocity%7D-object_%7Bvelocity%7D;%20contact%5D,%20%5Cend%7Bequation%7D)

where hand<sub>joints</sub> and hand<sub>velocities</sub> denotes joint positions (angles) and joint velocities (angles) respectively. 
hand<sub>pos</sub> - object<sub>pos</sub> denotes the relative position between the palm of the hand and the object (6D difference in position and orientation) and 
hand<sub>velocity</sub> - object<sub>velocity</sub> the same but in velocities. contact measures the distance between fingertips and object surface up to 4 cm. The total dimension of the state vector is 63.

### Action space

The action space consists of the target position of the joints (in angles) and the differential of translation and rotation relative to previous timestep. 
There are 23 actuators controlling hand joints and 6 actuators for global translation and rotation of the hand (3D position + 3D rotation).

### Hand Environment for Control
The mujoco environment that produced the states and actions for the dataset can be found in the "robot_hand/envs/hand_env_sphere.py" file. 
It can be used with various control methodologies, such as imitation or reinforcement learning.