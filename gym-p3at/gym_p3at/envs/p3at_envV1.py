#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from itertools import product
from std_srvs.srv import Trigger, TriggerRequest, Empty
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
import message_filters
from nav_msgs.msg import Odometry
from math import sqrt
import time

class p3atEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    
    self.crash_robot = False
    self.pose_x = 0
    self.pose_y = 0
    self.pose_goal_x = 6.5
    self.pose_goal_y = 2.5
    self.list_poses = []

    self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(5,), dtype=np.float32)
    
    #self.action_space = spaces.Dict({
    #        "times":spaces.Box(low=1, high=5, shape=(1,), dtype=np.int32),
    #        "x":spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
    #        "z":spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    #    })

    self.action_space = spaces.Box(np.array([-0.2,-1]),np.array([1,1]),dtype=np.float32) #Action space continuous

    ################### Space Action GRID (Discret) #############
    
    #self.time_sec=np.linspace(1, 5, 5, dtype = np.int32)
    #self.x=np.linspace(-0.2, 1, 20, dtype = np.float32)
    #self.z=np.linspace(-1, 1, 20, dtype = np.float32)

    #self.action_space = spaces.Discrete(400)

    #mapping = [self.time_sec,self.x,self.z]
    #mapping = [self.x,self.z]
    #self.actions=list(product(*mapping))
    
    #############################################################

    ################### eight moves ###########################
    #self.x=[0.15, -0.15, 0, 0, 0.15, 0,15, -0,15, -0.15]
    #self.z=[0, 0, 0.15, -0.15, 0.15, -0,15, 0,15, -0.15]

    #self.action_space = spaces.Discrete(8)

    #print(self.actions, len(self.actions))

    rospy.init_node('p3at_env')

    # wait for this service to be running
    # Ideally, this service should run 24/7, but remember it's fake :) 
    rospy.wait_for_service('/array_laser')

    # Create the connection to the service. Remeber it's a Trigger service
    self.sonar_service = rospy.ServiceProxy('/array_laser', Trigger)
    self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)

    # Create an object of type TriggerRequest. We need a TriggerRequest for a Trigger service
    # We don't need to pass any argument because it doesn't take any
    self.sonar_response = TriggerRequest()

    ############ Subscribes ##############
    rospy.Subscriber("/sim_p3at/bumper_states", ContactsState, self.callback, 1)
    rospy.Subscriber("/sim_p3at/pose", Odometry, self.callback, 2)

    ############### Publishes ##################
    self.cmd_vel = rospy.Publisher('/sim_p3at/cmd_vel', Twist, queue_size=2)

    print("Init Environment P3AT 1 (Continuous)")

  def callback(self, data, code):
    #print(ContactsState.states)
    if code==1:
      if(data.states):
        self.crash_robot=True
      else:
        self.crash_robot=False
    else:
      self.pose_x=data.pose.pose.position.x
      self.pose_y=data.pose.pose.position.y

  def step(self, action):
    #print(action[0], action[1], action[2])

    crash, dist=self.move(action)

    ob=self.get_observation()

    reward = 0

    radius = 0.3

    if crash==True:
      done=True
      reward=-100
    else:
      done=False

      if dist>0:#0.05:
        reward=dist
      elif dist==0:
        reward=-5

      if (self.pose_goal_x-self.pose_x)**2 + (self.pose_goal_y-self.pose_y)**2 <= 1**2:
        done=True
        reward=500
        self.stop_robot()
        self.reset()

      for i, x in enumerate(self.list_poses):
        if i<len(self.list_poses)-5:
          if (x[0]-self.pose_x)**2 + (x[1]-self.pose_y)**2 <= radius**2:
            print("Visited!!!")
            reward=-30

    return ob, reward, done, {}

  def move(self, action):
    twist = Twist()
    twist.linear.x = action[0][0]
    twist.angular.z = action[0][1]

    #twist.linear.x = self.x[action]
    #twist.angular.z = self.z[action]

    crash=False

    execution_time = 1#self.actions[action][0]

    now = rospy.get_rostime()

    #seconds = rospy.get_time()

    #print(seconds)

    xA = self.pose_x
    yA = self.pose_y

    self.list_poses.append([xA,yA])
    #rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

    last_time = execution_time + now.secs

    while(last_time>now.secs):
      self.cmd_vel.publish(twist)
      now = rospy.get_rostime()
      #rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
      #print(twist)
      if self.crash_robot==True:
        crash=True
        self.stop_robot()
        self.reset()
        break

    xB = self.pose_x
    yB = self.pose_y

    dist=self.distance(xA,xB,yA,yB)

    if action[0][0]<=0: #favorecer o robo ir pra frente
      dist=0

    self.stop_robot()
    return (crash, dist)

  def distance(self, xA, xB, yA, yB):
    return(sqrt((xA-xB)**2) + ((yA-yB)**2))

  def stop_robot(self):
    twist = Twist()
    twist.linear.x = 0
    twist.angular.z = 0

    self.cmd_vel.publish(twist)

  def get_observation(self):
    # Now send the request through the connection
    result = self.sonar_service(self.sonar_response)


    res = result.message.strip('][').split(', ')

    result = list(map(lambda x: float(x.replace(",", "")), res))
    
    #print(result)
    return np.asarray(result)

  def reset(self):
    self.reset_world()
    self.list_poses.clear()
    time.sleep(1)
    obs=self.get_observation()

    return obs

  def render(self, mode='human', close=False):
    pass