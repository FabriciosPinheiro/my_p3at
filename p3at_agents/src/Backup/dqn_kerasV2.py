#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import gym_arduino

EPISODES = 1000

class DQNAgent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=2000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.tau = .125
		self.epsilon_decay = 0.0005#0.995
		self.learning_rate = 0.01#0.001
		self.model = self._build_model()
		self.target_model = self._build_model()

	def _build_model(self):
		# Neural Net for Deep-Q learning Model
		model = Sequential()
		model.add(Dense(32, input_dim=self.state_size, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model

	def memorize(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		#if self.epsilon > self.epsilon_min:
		#	self.epsilon *= self.epsilon_decay
		if np.random.rand() < self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])  # returns action

	'''
	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = (reward + self.gamma *
						  np.amax(self.model.predict(next_state)[0]))
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)
		#if self.epsilon > self.epsilon_min:
			#self.epsilon *= self.epsilon_decay
	'''
	def replay(self, batch_size):
		samples = random.sample(self.memory, batch_size)
		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.target_model.predict(state)
			if done:
				target[0][action] = reward
			else:
				Q_future = max(self.target_model.predict(new_state)[0])
				target[0][action] = reward + Q_future * self.gamma
			self.model.fit(state, target, epochs=1, verbose=0)

	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
		self.target_model.set_weights(target_weights)

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)

def smooth(x):
	# last 100
	n = len(x)
	y = np.zeros(n)
	for i in range(n):
		start = max(0, i - 99)
		y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
	return y

if __name__ == "__main__":
	env = gym.make('arduino-v0')
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n
	agent = DQNAgent(state_size, action_size)
	# agent.load("arduino-dqn.h5")
	done = False
	batch_size = 32

	reward_list=[]

	env.reset()

	#plt.title('Learing')
	#plt.xlabel('Episode')
	#plt.ylabel('Reward')


	fig = plt.figure()
	fig.set_size_inches(20,6)
	ax = fig.add_subplot(1, 1, 1)
	ay = fig.add_subplot(1, 1, 1)

	for e in range(EPISODES):
		state = env.get_observation()
		state = np.reshape(state, [1, state_size])
		reward_total=0
		for time in range(20):
			# env.render()
			action = agent.act(state)
			next_state, reward, done, _ = env.step(action)

			print("Step: [%s] - Action: [%.2f %.2f] \t| State: %s \t| reward: %.4f \t| done: %s" % 
									(time, env.actions[action][0], env.actions[action][1], next_state, reward, done))

			next_state = np.reshape(next_state, [1, state_size])
			agent.memorize(state, action, reward, next_state, done)
			state = next_state

			reward = reward if not done else -10

			reward_total+=reward
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size)

		if e % 20 == 0:
			agent.target_train()

		agent.epsilon = agent.epsilon_min + (agent.epsilon - agent.epsilon_min) * np.exp(-agent.epsilon_decay * e)

		reward_list.append(reward_total)

		#For real time
		x = np.array(reward_list)
		y = smooth(x)
		ax.clear()
		ay.clear()
		ax.plot(x, label='orig')
		ay.plot(y, label='smoothed')
		plt.legend()
		plt.savefig("graph.png")
	  
		print("episode: {}/{}, score: {}, e: {:.3}"
			  .format(e, EPISODES, reward_total, agent.epsilon))

		if e % 100 == 0:
			 agent.save("arduino-dqn.h5")