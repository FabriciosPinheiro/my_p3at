#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import sys
import gym
import gym_p3at
import pylab
import numpy as np
import random
import os.path
import time
import tensorflow as tf
from datetime import datetime
from collections import deque
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow import keras

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

        agent.train_loss = np.mean(agent.train_losses)
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

# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.995
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01

        self.batch_size = 32
        self.train_start = 1000

        path = os.getcwd()
        date=self.date_time()

        # create replay memory using deque
        self.memory = deque(maxlen=50000)
        self.train_losses = []
        self.train_loss = 0
        self.advantages = 0
        self.max_mean = 0

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.critic_target = self.build_critic()

        self.experiment = "complete"

        if self.load_model:
            actor_file = "./save_model/p3at_actor_"+self.experiment+".h5"
            #critic_file = "./save_model/p3at_critic_"+self.experiment+".h5"
            self.epsilon = self.epsilon_min
            if os.path.exists(actor_file):
                self.actor.load_weights(actor_file)
                #self.critic.load_weights("./save_model/p3at_critic_"+self.experiment+".h5")
                self.sess = tf.compat.v1.Session()
                self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
                self.summary_writer = tf.compat.v1.summary.FileWriter("./graph/p3at_v0-a2c-"+self.experiment+"-"+date, self.sess.graph)

                self.sess.run(tf.global_variables_initializer())
            else:
                print("File does not exists!")
                print(file)
                sys.exit()
        else:
            self.epsilon = 1
            self.sess = tf.compat.v1.InteractiveSession()
            self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
            self.summary_writer = tf.compat.v1.summary.FileWriter("./graph/p3at_v0-a2c-"+self.experiment+"-"+date, self.sess.graph)

            self.sess.run(tf.global_variables_initializer())

    def date_time(self):
        date_hour = datetime.now()
        return (date_hour.strftime("%d_%m_%Y_%H_%M"))

    def setup_summary(self):
        name = "p3at-v0"
        episode_total_reward = tf.Variable(0.)
        tf.compat.v1.summary.scalar(name + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.compat.v1.summary.scalar(name + '/Average Advantage/Episode', episode_avg_max_q)
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

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(256, input_dim=self.state_size, activation='relu'))
        actor.add(Dense(256, activation='relu'))
        actor.add(Dense(self.action_size, activation='softmax'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.actor_lr, clipnorm=1.0))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(256, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(256, activation='relu'))
        critic.add(Dense(self.value_size, activation='linear'))
        critic.summary()
        critic.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(lr=self.critic_lr, clipnorm=1.0))
        return critic

    def update_target_model(self):
        self.critic_target.set_weights(self.critic.get_weights())

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            policy = self.actor.predict(state, batch_size=1).flatten()
            #return np.argmax(policy)
            return np.random.choice(self.action_size, 1, p=policy)[0]

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # update policy network every episode
    def train_model(self):
        if (len(self.memory) < self.train_start) or (self.load_model):
            return

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        target = np.zeros((batch_size, self.value_size))
        advantages = np.zeros((batch_size, self.action_size))

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        value, next_value = [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        for i in range(batch_size):
            value.append(self.critic.predict(update_input)[0])
            next_value.append(self.critic.predict(update_target)[0])

        #print(value, next_value)

        for i in range(self.batch_size):
            if done[i]:
                advantages[i][action[i]] = reward[i] - value[i]
                target[i][0] = reward[i]
            else:
                advantages[i][action[i]] = reward[i] + self.discount_factor * (next_value[i]) - value[i]
                target[i][0] = reward[i] + self.discount_factor * next_value[i]

        self.advantages = np.mean(advantages)

        self.actor.fit(update_input, advantages, epochs=1, verbose=0, callbacks=[EarlyStoppingAtMinLoss()])
        #self.critic.fit(update_input, target, epochs=1, verbose=0)
        self.critic_target.fit(update_input, target, epochs=1, verbose=0)

    '''
    def train_model(self, state, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        print(target, advantages)

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        print(value, next_value)

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)
    '''
if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('p3at-v0')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    env.reset(reset_type=False)

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.get_observation() #env.reset()
        state = np.reshape(state, [1, state_size])

        time = 0
        e += 1

        while not done:
            if agent.render:
                env.render()

            time += 1

            #if agent.epsilon == 1:
            #    reset_type = True
            #else:
            #    reset_type = False

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action, reset_type=False)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100

            if not done and reward>0:
                reward = (reward+1) * time

            print("Episode: [%s] \t| Step: [%s] - Action: [%.2f %.2f] \t| State: %s \t| reward: %.4f \t| done: %s" % 
                                    (e, time, env.actions[action][0], env.actions[action][1], next_state, reward, done))

            agent.append_sample(state, action, reward, next_state, done)

            #agent.train_model(state, action, reward, next_state, done)
 
            agent.train_model()

            score += reward
            state = next_state

            if agent.load_model:
                stats = [score, agent.advantages,
                    time, agent.train_loss/e, agent.epsilon]
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={
                        agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, time + 1)

            if done:
                # every episode, plot the play time
                scores.append(score)
                #episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("./save_graph/p3at_a2c_"+agent.experiment+".png")
                mean_last_20 = np.mean(scores[-min(50, len(scores)):])
                print("episode:", e, "\tscore:", score, "\tepsilon:", agent.epsilon, "\tmemory: ", len(agent.memory), "\tmean_last_50: ", mean_last_20)

                if not agent.load_model:
                    stats = [score, agent.advantages,
                        time, agent.train_loss/e, agent.epsilon]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                            })
                        summary_str = agent.sess.run(agent.summary_op)
                        agent.summary_writer.add_summary(summary_str, e)

                    #if agent.epsilon<agent.epsilon_min and mean_last_20 < 100:
                    #    agent.epsilon = 0.5

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if mean_last_20 > 4000:
                        agent.update_target_model()
                        agent.actor.save_weights("./save_model/p3at_actor_"+agent.experiment+".h5")
                        agent.critic.save_weights("./save_model/p3at_critic_"+agent.experiment+".h5")
                        sys.exit()

            if done and agent.load_model:
                print("Stop execution")
                sys.exit()

        # save the model
        if mean_last_20>agent.max_mean and not agent.load_model:
            agent.max_mean=mean_last_20
            agent.update_target_model()
            agent.actor.save_weights("./save_model/p3at_actor_"+agent.experiment+"_"+str(round(mean_last_20,2))+"["+str(e)+"].h5")
            agent.critic.save_weights("./save_model/p3at_critic_"+agent.experiment+"_"+str(round(mean_last_20,2))+"["+str(e)+"].h5")

        if (e % 10 == 0) and not agent.load_model:
            agent.update_target_model()
            agent.actor.save_weights("./save_model/p3at_actor_"+agent.experiment+"_last_model.h5")
            agent.critic.save_weights("./save_model/p3at_critic_"+agent.experiment+"_last_model.h5")