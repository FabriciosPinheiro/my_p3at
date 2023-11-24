import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym import wrappers


class PancamEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self, goal_velocity = 0):
    self.min_action = -1.0
    self.max_action = 1.0
    self.min_position = -0.7
    self.max_position = 0.7
    self.max_speed = 0.07
    self.goal_position = 0.45 # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
    self.goal_velocity = goal_velocity
    self.power = 0.0015

    self.low_state = np.array([self.min_position, -self.max_speed])
    self.high_state = np.array([self.max_position, self.max_speed])

    self.viewer = None

    self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                   shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

    self.seed()

    print ("Inited !")

  def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

  def step(self, action):
    pass
    

    #print(go.x)
    #result = go.movebase_client()
    #if result:
    #  print("Goal execution done!")

    """

    Parameters
    ----------
    action :

    Returns
    -------
    ob, reward, episode_over, info : tuple
        ob (object) :
            an environment-specific object representing your observation of
            the environment.
        reward (float) :
            amount of reward achieved by the previous action. The scale
            varies between environments, but the goal is always to increase
            your total reward.
        episode_over (bool) :
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
        info (dict) :
             diagnostic information useful for debugging. It can sometimes
             be useful for learning (for example, it might contain the raw
             probabilities behind the environment's last state change).
             However, official evaluations of your agent are not allowed to
             use this for learning.
  
    self._take_action(action)
    self.status = self.env.step()
    reward = self._get_reward()
    ob = self.env.getState()
    episode_over = self.status != hfo_py.IN_GAME
    return ob, reward, episode_over, {}
    """

  def reset(self):
    self.state = self.np_random.uniform(low=0, high=10, size=(270,))
    self.steps_beyond_done = None
    return np.array(self.state)

  def render(self, mode='human', close=False):
    pass

'''
if __name__ == '__main__':
  teste = PathEnv()
  print(teste.observation_space)
  '''