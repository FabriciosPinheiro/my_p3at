#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import gym
import numpy as np
from itertools import product
import gym_arduino

env = gym.make('arduino-v0')
#env.reset()

#print(env.get_observation())

value=env.action_space.sample()
print(env.actions[value])
#print(env.action_space.shape)

#print(env.observation_space.shape[0])
'''
for i_episode in range(5):
	observation = env.reset()
	for t in range(5):
		#env.render()
		#print(observation)
		action = env.action_space.sample()
		observation, reward, done, info = env.step(action)
		print(action, observation, reward, done)

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break
env.close()
'''