#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import sys
import gym
import matplotlib.pyplot as plt
import random
import numpy as np
import gym_p3at
import os.path
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from datetime import datetime
import tensorflow as tf

EPISODES = 10000


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
	def __init__(self, state_size, action_size):
		# if you want to see Cartpole learning, then change to True
		self.render = False
		self.load_model = True
		# get size of state and action
		self.state_size = state_size
		self.action_size = action_size

		# these is hyper parameters for the Double DQN
		self.discount_factor = 0.99
		self.learning_rate = 0.001

		if self.load_model:
			self.epsilon = 0.01
		else:
			self.epsilon = 1.0

		self.epsilon_decay = 0.999
		self.epsilon_min = 0.01
		self.batch_size = 32
		self.train_start = 200

		path = os.getcwd()

		file = path+"/arduino_23_10_2020_23_08.h5"
		# create replay memory using deque
		self.memory = deque(maxlen=2000)

		self.sess = tf.compat.v1.InteractiveSession()
		self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
		self.summary_writer = tf.compat.v1.summary.FileWriter("./graph", self.sess.graph)

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()

		# initialize target model
		self.update_target_model()

		if self.load_model:
			#print('getcwd:      ', os.getcwd())
			if os.path.exists(file):
				self.model.load_weights(file)
				self.update_target_model()
			else:
				print("File does not exists!")
				print(file)
				sys.exit()

	def setup_summary(self):
		name = "p3at-v0"
		episode_total_reward = tf.Variable(0.)
		tf.compat.v1.summary.scalar(name + '/Total Reward/Episode', episode_total_reward)
		episode_avg_max_q = tf.Variable(0.)
		tf.compat.v1.summary.scalar(name + '/Average Max Q/Episode', episode_avg_max_q)
		episode_duration = tf.Variable(0.)
		tf.compat.v1.summary.scalar(name + '/Duration/Episode', episode_duration)
		episode_avg_loss = tf.Variable(0.)
		tf.compat.v1.summary.scalar(name + '/Average Loss/Episode', episode_avg_loss)
		episode_epsilon = tf.Variable(0.)
		tf.compat.v1.summary.scalar(name + '/Epsilon/Episode', episode_epsilon)
		summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss, episode_epsilon]
		summary_placeholders = [tf.compat.v1.placeholder(tf.float32) for _ in range(len(summary_vars))]
		update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
		summary_op = tf.compat.v1.summary.merge_all()
		return summary_placeholders, update_ops, summary_op

	# approximate Q function using Neural Network
	# state is input and Q Value of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Dense(256, input_dim=self.state_size, activation='relu',
						kernel_initializer='he_uniform'))
		model.add(Dense(256, activation='relu',
						kernel_initializer='he_uniform'))
		model.add(Dense(self.action_size, activation='linear',
						kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(state)
			return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
		if len(self.memory) < self.train_start:
			return
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if (len(self.memory) < self.train_start) or (self.load_model):
			return
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)

		update_input = np.zeros((batch_size, self.state_size))
		update_target = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(batch_size):
			update_input[i] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			update_target[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		target = self.model.predict(update_input)
		target_next = self.model.predict(update_target)
		target_val = self.target_model.predict(update_target)

		for i in range(self.batch_size):
			# like Q Learning, get maximum Q value at s'
			# But from target model
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				# the key point of Double DQN
				# selection of action is from model
				# update is from target model
				a = np.argmax(target_next[i])
				target[i][action[i]] = reward[i] + self.discount_factor * (
					target_val[i][a])

		# make minibatch which includes target q value and predicted q value
		# and do the model fit!
		self.model.fit(update_input, target, batch_size=self.batch_size,
					   epochs=1, verbose=0)

def smooth(x):
	# last 100
	n = len(x)
	y = np.zeros(n)
	for i in range(n):
		start = max(0, i - 99)
		y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
	return y

def date_time():
	date_hour = datetime.now()
	return (date_hour.strftime("%d_%m_%Y_%H_%M"))

if __name__ == "__main__":
	# In case of CartPole-v1, you can play until 500 time step
	env = gym.make('pat-v0')
	# get size of state and action from environment
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	print(state_size)

	agent = DoubleDQNAgent(state_size, action_size)

	reward_list=[]
	date=date_time()

	env.reset()


	fig = plt.figure()
	fig.set_size_inches(20,6)
	ax = fig.add_subplot(1, 1, 1)
	ay = fig.add_subplot(1, 1, 1)
	
	for e in range(EPISODES):
		done = False
		state = env.get_observation() #env.reset()
		state = np.reshape(state, [1, state_size])
		reward_total=0
		time=0
		while not done:
			if agent.render:
				env.render()

			# get action for the current state and go one step in environment
			action = agent.get_action(state)
			next_state, reward, done, info = env.step(action)
			next_state = np.reshape(next_state, [1, state_size])

			#if time==19: done=True
			if not done and reward>0:
				reward = reward * time
			else: 
				reward = -1


			print("Step: [%s] - Action: [%.2f %.2f] \t| State: %s \t| reward: %.4f \t| done: %s" % 
									(time, env.actions[action][0], env.actions[action][1], next_state, reward, done))

			time+=1
			# if an action make the episode end, then gives penalty of -100

			# save the sample <s, a, r, s'> to the replay memory
			agent.append_sample(state, action, reward, next_state, done)
			# every time step do the training
			agent.train_model()
			reward_total+=reward
			state = next_state

			if done and not agent.load_model:
				reward_list.append(reward_total)

				stats = [reward_total, 0,
					time, 0, agent.epsilon]
				for i in range(len(stats)):
					agent.sess.run(agent.update_ops[i], feed_dict={
						agent.summary_placeholders[i]: float(stats[i])
						})
					summary_str = agent.sess.run(agent.summary_op)
					agent.summary_writer.add_summary(summary_str, EPISODES + 1)

				#For real time
				x = np.array(reward_list)
				y = smooth(x)
				ax.clear()
				ay.clear()
				ax.plot(x, label='orig')
				ay.plot(y, label='smoothed')
				plt.legend()
				plt.savefig("graph"+date+".png")

				print("episode:", e, "  score:", reward_total, "  memory length:",
					  len(agent.memory), "  epsilon:", agent.epsilon)

				# if the mean of scores of last 10 episode is bigger than 490
				# stop training
				#if np.mean(scores[-min(10, len(scores)):]) > 490:
				 #   sys.exit()

		# save the model
		if (e % 10 == 0) and not agent.load_model:
			# every episode update the target model to be same with model
			agent.update_target_model()
			agent.model.save_weights("arduino_"+date+".h5")