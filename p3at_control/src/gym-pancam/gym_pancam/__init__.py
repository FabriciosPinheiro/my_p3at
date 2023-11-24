from gym.envs.registration import register

register(
    id='pancam-v0',
    entry_point='gym_pancam.envs:PancamEnv',
)