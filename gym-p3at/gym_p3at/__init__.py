from gym.envs.registration import register

register(
    id='p3at-v0',
    entry_point='gym_p3at.envs:p3atEnv',
)

register(
    id='p3at-v1',
    entry_point='gym_p3at.envs.p3at_envV1:p3atEnv',
)

register(
    id='p3at-v2',
    entry_point='gym_p3at.envs.p3at_envV2:p3atEnv',
)