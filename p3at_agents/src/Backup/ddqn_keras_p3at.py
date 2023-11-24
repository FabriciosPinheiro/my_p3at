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
from datetime import datetime
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

#For sounds
from pydub import AudioSegment
from pydub.playback import play


EPISODES = 20000

class EarlyStoppingAtMinLoss(keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
    	'''
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf
		'''

    def on_epoch_end(self, epoch, logs=None):
        agent.train_losses = logs.get("loss")

        agent.train_loss = agent.train_losses
        '''
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
		'''
    def on_train_end(self, logs=None):
        pass
        #if self.stopped_epoch > 0:
        #    print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


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
		self.discount_factor = 0.95
		self.learning_rate = 0.001

		self.epsilon_decay = 0.999
		self.epsilon_min = 0.1
		self.batch_size = 64
		self.train_start = 1000

		path = os.getcwd()
		date=self.date_time()

		# create replay memory using deque
		self.memory = deque(maxlen=50000)
		self.train_losses = []
		self.train_loss = 0
		self.list_max_q = []
		self.max_q = 0
		self.max_mean = 0

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()

		self.experiment = "exp_05-real-10"

		#sounds
		self.sound = AudioSegment.from_file("finished.wav")
		self.sound_crash = AudioSegment.from_file("car-crash.wav")

		if self.load_model:
			#file = "./save_model/result_ok/DQN/p3at_dqn_"+self.experiment+".h5"
			#file = "./save_model/partial/p3at_dqn_complete_limit_3699.94.h5"
			#file = "./save_model/partial/p3at_dqn_complete_limit_3878.45.h5"
			#file = "./save_model/partial/p3at_dqn_complete_limit_4683.6.h5"
			file = "./save_model/partial/p3at_dqn_complete_limit_4582.76.h5"
			self.epsilon = self.epsilon_min
			if os.path.exists(file):
				self.model.load_weights(file)
				self.update_target_model()
				self.sess = tf.compat.v1.Session()
				self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
				self.summary_writer = tf.compat.v1.summary.FileWriter("./graph/p3at_v0-dqn-result-"+self.experiment+"-"+date, self.sess.graph)

				self.sess.run(tf.global_variables_initializer())
			else:
				print("File does not exists!")
				print(file)
				sys.exit()
		else:
			self.epsilon = 1
			# initialize target model
			self.update_target_model()
			self.sess = tf.compat.v1.InteractiveSession()
			self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
			self.summary_writer = tf.compat.v1.summary.FileWriter("./graph/p3at_v0-dqn-train"+self.experiment+"-"+date, self.sess.graph)

			self.sess.run(tf.global_variables_initializer())

	def date_time(self):
		date_hour = datetime.now()
		return (date_hour.strftime("%d_%m_%Y_%H_%M"))

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
		model.add(Dense(256, input_dim=self.state_size, activation='relu'))
		model.add(Dense(256, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.summary()
		model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.learning_rate, clipnorm=1.0))
		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		q_value = self.model.predict(state)
		predict = np.argmax(q_value[0])
		
		self.list_max_q.append(np.amax(q_value))
		self.max_q = np.mean(self.list_max_q)

		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			print("Predict", predict)
			return predict

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
					   epochs=1, verbose=0, callbacks=[EarlyStoppingAtMinLoss()])

if __name__ == "__main__":
	# In case of CartPole-v1, you can play until 500 time step
	env = gym.make('p3at-v0')
	# get size of state and action from environment
	state_size = env.observation_space.shape[0]
	action_size = env.action_space.n

	#print(state_size)

	agent = DoubleDQNAgent(state_size, action_size)

	reward_list=[]

	env.reset(reset_type=False)
	
	for e in range(EPISODES):
		done = False
		state = env.get_observation() #env.reset()
		state = np.reshape(state, [1, state_size])
		reward_total=0
		time=0
		mean_last_50 = 0
		e += 1
		while not done:
			if agent.render:
				env.render()

			time+=1

			# get action for the current state and go one step in environment
			action = agent.get_action(state)
			next_state, reward, done, info = env.step(action, reset_type=False)
			next_state = np.reshape(next_state, [1, state_size])

			if not done and reward>0:
				reward = (reward+1) * time

			print("Episode: [%s] \t| Step: [%s] - Action: [%.2f %.2f] \t| State: %s \t| reward: %.4f \t| done: %s" % 
									(e, time, env.actions[action][0], env.actions[action][1], next_state, reward, done))
			
			# if an action make the episode end, then gives penalty of -100

			# save the sample <s, a, r, s'> to the replay memory
			agent.append_sample(state, action, reward, next_state, done)
			# every time step do the training
			agent.train_model()
			reward_total+=reward
			state = next_state

			if agent.load_model:
				stats = [reward_total, agent.max_q,
					time, agent.train_loss/e, agent.epsilon]
				for i in range(len(stats)):
					agent.sess.run(agent.update_ops[i], feed_dict={
						agent.summary_placeholders[i]: float(stats[i])
						})
					summary_str = agent.sess.run(agent.summary_op)
					agent.summary_writer.add_summary(summary_str, time + 1)

			if done:
				reward_list.append(reward_total)

				mean_last_50 = np.mean(reward_list[-min(50, len(reward_list)):])
				print("episode:", e, "\tscore:", reward_total, "\tepsilon:", agent.epsilon, "\tmemory: ", len(agent.memory), "\tmean_last_50: ", mean_last_50)

				if not agent.load_model:
					stats = [reward_total, agent.max_q,
						time, agent.train_loss/e, agent.epsilon]
					for i in range(len(stats)):
						agent.sess.run(agent.update_ops[i], feed_dict={
							agent.summary_placeholders[i]: float(stats[i])
							})
						summary_str = agent.sess.run(agent.summary_op)
						agent.summary_writer.add_summary(summary_str, e)

					# if the mean of scores of last 10 episode is bigger than 490
					# stop training
					if mean_last_50 > 5000:
						agent.update_target_model()
						agent.model.save_weights("./save_model/partial/p3at_dqn_"+agent.experiment+"_"+str(round(mean_last_50,2))+".h5")
						sys.exit()

			if done and agent.load_model:
				print("Stop execution")
				env.save_metrics(agent.experiment)
				if env.failure==0:
					play(agent.sound)
					print("Successfully!!!")
				elif env.failure==1:
					play(agent.sound_crash)
					print("Crashed!!!")
				sys.exit()

		# save the model
		if mean_last_50>agent.max_mean and not agent.load_model:
			# every episode update the target model to be same with model
			agent.max_mean=mean_last_50
			agent.update_target_model()
			agent.model.save_weights("./save_model/partial/p3at_dqn_"+agent.experiment+"_"+str(round(mean_last_50,2))+"["+str(e)+"].h5")

		if (e % 10 == 0) and not agent.load_model:
			agent.update_target_model()
			agent.model.save_weights("./save_model/partial/p3at_dqn_"+agent.experiment+"_last_model.h5")