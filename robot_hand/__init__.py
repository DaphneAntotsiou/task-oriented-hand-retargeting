from gym.envs.registration import register

register(
    id='hand_sphere-v0',
    entry_point='robot_hand.envs:HandEnvSphere',
    timestep_limit=150,
    reward_threshold=5e50,
    # nondeterministic=True,
)