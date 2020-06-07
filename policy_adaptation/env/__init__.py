from gym.envs.registration import register


register(
    id='HalfCheetahDM-v2',
    entry_point='env.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='AntDM-v2',
    entry_point='env.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
