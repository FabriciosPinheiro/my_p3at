#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from itertools import product
from std_srvs.srv import Trigger, TriggerRequest, Empty
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ContactsState, ModelState
import message_filters
from nav_msgs.msg import Odometry
from math import sqrt
import math
from squaternion import Quaternion
import time
from signal import signal, SIGINT
from sys import exit
from time import strftime, gmtime

class p3atEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    
    self.crash_robot = False
    self.pose_x = 0
    self.pose_y = 0
    self.pose_goal_x = 5.7 #2.0
    self.pose_goal_y = 1.8 #-5.8
    self.list_poses = []
    self.max_steps = 0
    self.laser_full = []
    self.security_layer = True
    self.security_distance = 0.30 #Real = 0.30
    self.simulation = False
    self.distance_to_go = 100
    self.list_path = []
    self.list_velocity = []
    self.traveled_distance = 0
    self.failure = -1
    self.count_security = 0
    self.touch = False
    self.max_touch_step = 5

    #self.start_list = [[0.5,0.5,0],[2.5,3.5,180],[0.5,4.5,90],[3.5,2.5,-90],[4.5,3.5,90]]

    self.observation_space = spaces.Box(low=0.0, high=6.0, shape=(10,), dtype=np.float32)
    
    #self.action_space = spaces.Dict({
    #        "times":spaces.Box(low=1, high=5, shape=(1,), dtype=np.int32),
    #        "x":spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
    #        "z":spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    #    })

    #self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]),dtype=np.float32) #Action space continuous

    ################### Space Action GRID (Discret) #############
    #model 100
    self.x=np.array([-0.1, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    self.z=np.linspace(-0.5, 0.5, 10, dtype = np.float32)

    self.action_space = spaces.Discrete(100)

    #Model 400
    #self.x=np.linspace(-0.2, 1, 20, dtype = np.float32)
    #self.z=np.linspace(-1, 1, 20, dtype = np.float32)

    #self.action_space = spaces.Discrete(400)
    #print(self.x)
    #print(self.z)

    #exit()

    #mapping = [self.time_sec,self.x,self.z]
    mapping = [self.x,self.z]
    self.actions=list(product(*mapping))
    
    #############################################################

    ################### eight moves ###########################
    #self.x=[0.15, -0.15, 0, 0, 0.15, 0,15, -0,15, -0.15]
    #self.z=[0, 0, 0.15, -0.15, 0.15, -0,15, 0,15, -0.15]

    #self.action_space = spaces.Discrete(8)

    #print(self.actions, len(self.actions))

    rospy.init_node('p3at_env')

    self.mission_duration = rospy.Time.now().secs

    # wait for this service to be running
    # Ideally, this service should run 24/7, but remember it's fake :) 
    rospy.wait_for_service('/array_laser')

    # Create an object of type TriggerRequest. We need a TriggerRequest for a Trigger service
    # We don't need to pass any argument because it doesn't take any
    self.sonar_response = TriggerRequest()

    # Create the connection to the service. Remeber it's a Trigger service
    self.sonar_service = rospy.ServiceProxy('/array_laser', Trigger)
    if self.simulation:
      self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty) #For simulation
      self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState) #For simulation

      ############ Subscribes ##############
      rospy.Subscriber("/sim_p3at/bumper_states", ContactsState, self.callback, 1)
      rospy.Subscriber("/sim_p3at/pose", Odometry, self.callback, 2)
      rospy.Subscriber("/sim_p3at/scan", LaserScan, self.callback, 3)

      ############### Publishes ##################
      self.cmd_vel = rospy.Publisher('/sim_p3at/cmd_vel', Twist, queue_size=10)
    else:
      ########## Subscribes RosAria ##############
      rospy.Subscriber("/rosaria/bumper_states", ContactsState, self.callback, 1)
      rospy.Subscriber("/rosaria/pose", Odometry, self.callback, 2)
      rospy.Subscriber("/scan", LaserScan, self.callback, 3)

      ############### Publishes RosAria ##################
      self.cmd_vel = rospy.Publisher('/rosaria/cmd_vel', Twist, queue_size=10)

    signal(SIGINT, self.handler)

    print("Init Environment P3AT 0")

  def handler(self, signal_received, frame):
    # Handle any cleanup here
    for i in range(10):
      self.stop_robot()

    self.failure=1
    self.save_metrics("exp_05-real-10")
    print('SIGINT or CTRL-C detected. Stop robot!')
    time.sleep(1)
    exit(0)

  def callback(self, data, code):
    #print(ContactsState.states)
    if code==1:
      if(data.states):
        self.crash_robot=True
      else:
        self.crash_robot=False
    elif code==2:
      self.pose_x=data.pose.pose.position.x
      self.pose_y=data.pose.pose.position.y
      self.list_path.append([round(self.pose_x,2),round(self.pose_y,2)])
    elif code==3:
      self.laser_full = data.ranges

  def step(self, action, reset_type):
    #print(action[0], action[1], action[2])

    crash, dist=self.move(action)

    if self.touch and self.max_touch_step==5:
      self.count_security += 1
    
    self.max_touch_step -= 1

    if self.max_touch_step==0:
      self.max_touch_step=5

    #time.sleep(0.2)
    ob=self.get_observation()

    reward = 0

    radius = 0.2

    if crash==True or self.max_steps==0:
      done=True
      self.failure = 1
      reward=-100#10*(-self.max_steps)
      #self.reset(reset_type)
    else:
      done=False

      if dist>0:#0.05:
        reward=dist
      else:
        reward=-5

      if (self.pose_goal_x-self.pose_x)**2 + (self.pose_goal_y-self.pose_y)**2 <= 1**2:
        done=True
        self.failure = 0
        if reset_type==False:
          reward=10*self.max_steps
        self.stop_robot()
        self.reset(reset_type)

      for i in self.list_poses[-6::-1]:
        if (i[0]-self.pose_x)**2 + (i[1]-self.pose_y)**2 <= radius**2:
            print("Visited!!!")
            reward=-10
            break

      self.max_steps-=1

    return ob, reward, done, {}

  def move(self, action):
    self.touch = False

    twist = Twist()
    #if self.actions[action][0]>self.linear_limit_vel:
    #  twist.linear.x = self.linear_limit_vel
    #else:
    twist.linear.x = self.actions[action][0]
    twist.angular.z = self.actions[action][1]

    self.list_velocity.append(twist.linear.x)
    
    '''
    if vel_ang>0:
      if vel_ang>self.angular_limit_vel:
        twist.angular.z = self.angular_limit_vel
      else:
        twist.angular.z = vel_ang
    else:
      if vel_ang<-self.angular_limit_vel:
        twist.angular.z = -self.angular_limit_vel
      else:
        twist.angular.z = vel_ang
    '''

    crash=False

    execution_time = 0.2#self.actions[action][0]

    now = rospy.get_rostime()

    xA = self.pose_x
    yA = self.pose_y

    self.list_poses.append([xA,yA])
    #rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

    last_time = execution_time + now.secs

    while(last_time>now.secs):
       #Camada de segurança
      if self.security_layer:
        #print(len(self.laser_full))
        for i, x in enumerate(self.laser_full):
          if i>225 and i<315:
            if x**2 <= self.security_distance**2:
              print("TO BACK", i, x)
              self.stop_robot()
              twist = Twist()
              twist.linear.x = -0.1
              self.cmd_vel.publish(twist)
              self.touch = True
              break

          if i>135 and i<225:
            if x**2 <= self.security_distance**2:
              print("TO BACK", i, x)
              self.stop_robot()
              twist = Twist()
              twist.linear.x = -0.1
              twist.angular.z = 0.2
              self.cmd_vel.publish(twist)
              self.touch = True
              break

          if i>315 and i<405:
            if x**2 <= self.security_distance**2:
              print("TO BACK", i, x)
              self.stop_robot()
              twist = Twist()
              twist.linear.x = -0.1
              twist.angular.z = -0.2
              self.cmd_vel.publish(twist)
              self.touch = True
              break

      self.cmd_vel.publish(twist)
      now = rospy.get_rostime()
      #rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
      #print(twist)

      if self.crash_robot==True:
        crash=True
        self.stop_robot()
        self.reset(reset_type=False)
        break

    xB = self.pose_x
    yB = self.pose_y

    dist= self.distance(xA,xB,yA,yB)

    self.traveled_distance += abs(dist)

    print(self.count_security)

    '''
    #dist_goal_A=self.distance(self.pose_goal_x,xA,self.pose_goal_y,yA)
    dist_goal=self.distance(self.pose_goal_x,xB,self.pose_goal_y,yB)
    if dist_goal<self.distance_to_go:
      dist = dist+1
      self.distance_to_go = dist_goal
      print("Distance:", self.distance_to_go)
    else:
      dist = dist-0.5
    '''
    #if self.actions[action][0]<=0: #favorecer o robo ir pra frente
    #  dist=0

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
    
    #print("Laser: ",result)
    
    #print("Laser: ", self.laser_full)
    return np.asarray(result)

  def save_metrics(self, experiment):
    if self.list_velocity:
      f = open("./expLogsRobot/evaluation_"+experiment+".txt", "a")
      f.write("Experiment evaluation: "+experiment+"\n")
      
      f.write("Path: "+ str(self.list_path) + "\n")
      f.write("Average Velocity: "+ str(round(np.mean(self.list_velocity),2)) + "\n")
      f.write("Max Velocity: "+ str(round(np.max(self.list_velocity),2)) + "\n")

      f.write("Traveled distance: "+ str(round(self.traveled_distance,2)) + "\n")
      f.write("Count security: "+ str(self.count_security) + "\n")
      if self.failure==1:
        f.write("Mission: Failure!" + "\n")
      if self.failure==0:
        f.write("Mission: Success!" + "\n")
      #self.mission_duration = rospy.Time.now().secs-self.mission_duration
      #f.write("Mission Duration: "+ strftime("%H:%M:%S", gmtime(self.mission_duration)) + "\n")
      f.write("---------------------------------------------------\n")

      f.close()
    else:
      print("There are no parameters for evaluation!")

  def reset(self, reset_type):
    
    if reset_type==True:
      state_msg = ModelState()

      point = np.random.randint(len(self.start_list))

      direction = Quaternion.from_euler(0, 0, self.start_list[point][2], degrees=True)

      state_msg.model_name = "robot"
      state_msg.pose.position.x = self.start_list[point][0]
      state_msg.pose.position.y = self.start_list[point][1]
      state_msg.pose.position.z = 0
      state_msg.pose.orientation.x = 0
      state_msg.pose.orientation.y = 0
      state_msg.pose.orientation.z = direction[3]
      state_msg.pose.orientation.w = direction[0]
      self.set_model_state(state_msg)

    #else:
    #  self.reset_world()
      
    if self.max_steps==0:
      print("Max Steps 20")

    self.list_poses.clear()

    if self.simulation:
      self.max_steps=500
      self.reset_world()
    else:
      self.max_steps=500

    self.distance_to_go = 100
    time.sleep(0.2)
    obs=self.get_observation()

    return obs

  def render(self, mode='human', close=False):
    pass