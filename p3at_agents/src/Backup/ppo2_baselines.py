#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import gym
import gym_p3at
import tensorflow as tf

from stable_baselines import PPO2

# Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[512, 512])
# Create the agent
model = PPO2("MlpPolicy", "p3at-v0", tensorboard_log="./ppo2_tensorboard/", policy_kwargs=policy_kwargs, verbose=1)
# Retrieve the environment
env = model.get_env()
# Train the agent
model.learn(total_timesteps=100000)
# Save the agent
model.save("ppo2-p3at")

del model
# the policy_kwargs are automatically loaded
model = PPO2.load("ppo2-cartpole")