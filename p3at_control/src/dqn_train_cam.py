import gym
import gym_pancam
env = gym.make('pancam-v0')

print(env.action_space.sample())

print(env.observation_space)