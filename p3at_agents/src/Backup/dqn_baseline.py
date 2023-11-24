#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import gym
import gym_p3at

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

# Custom MLP policy of two layers of size 32 each
class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 256],
                                           layer_norm=False,
                                           feature_extraction="mlp")

env = gym.make('p3at-v0')

#model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./dqn_tensorboard/", n_cpu_tf_sess=1, double_q=False, 
#							learning_starts=200, policy_kwargs=dict(dueling=False))

model = DQN(CustomDQNPolicy, env, verbose=1, tensorboard_log="./dqn_tensorboard/", n_cpu_tf_sess=1, double_q=True, 
							learning_starts=500, learning_rate=0.001, buffer_size=10000, exploration_final_eps=0.01, 
							exploration_fraction=0.01, policy_kwargs=dict(dueling=True))
model.learn(total_timesteps=100000)
model.save("deepq_p3at")

model.summary()

#del model # remove to demonstrate saving and loading

#model = DQN.load("deepq_cartpole")

obs = env.reset()

time=0

for e in range(10000):
	time+=1
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)