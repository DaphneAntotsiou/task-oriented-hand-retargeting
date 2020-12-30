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


