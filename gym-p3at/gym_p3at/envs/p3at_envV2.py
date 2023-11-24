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
from squaternion import Quaternion
import time
from signal import signal, SIGINT

import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import os

class p3atEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    
    self.crash_robot = False
    self.pose_x = 0
    self.pose_y = 0
    self.max_steps = 0
    self.simulation = True
    self.failure = -1
    self.width = 64
    self.height = 64

    #self.cv_image = np.zeros((100,100,3), dtype=np.uint8)

    self.diff_left = 0
    self.diff_right = 0
    self.diff_top = 0
    self.diff_bottom = 0

    self.NMS_THRESHOLD=0.3
    self.MIN_CONFIDENCE=0.2

    self.person_center_x = 0
    self.person_center_y = 0
    self.green_x1 = 0
    self.green_y1 = 0

    self.green_x2 = 0
    self.green_y2 = 0

    self.blue_x1 = 0
    self.blue_y1 = 0
    self.blue_x2 = 0
    self.blue_y2 = 0

    self.check_reset = 0

    #self.observation_space = spaces.Box(low=0, high=255, shape=(self.width, self.height, 3), dtype=np.uint8) #For images

    self.observation_space = spaces.Box(low=-200.0, high=200.0, shape=(4,), dtype=np.float32)
  
    ################### Space Action GRID (Discret) #############
    
    #self.time_sec=np.linspace(1, 5, 5, dtype = np.int32)
    #self.x=np.array([-0.1, 0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    self.x=np.linspace(-0.2, 1.0, 10, dtype = np.float32)
    self.z=np.linspace(-0.5, 0.5, 10, dtype = np.float32)

    self.action_space = spaces.Discrete(100)

    mapping = [self.x,self.z]
    self.actions=list(product(*mapping))
    
    rospy.init_node('p3at_env')

    self.mission_duration = rospy.Time.now().secs

    self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    ############ Subscribes ##############
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/sim_p3at/camera/image_raw",Image,self.callback,queue_size=5)

    ############### Publishes ##################
    self.cmd_vel = rospy.Publisher('/sim_p3at/cmd_vel', Twist, queue_size=10)
    self.image_pub = rospy.Publisher("/sim_p3at/cam",Image)

    #rospy.spin()

    #Reconhecimento de pessoas
    self.dir = os.path.dirname(__file__)

    labelsPath = self.dir+"/coco.names"
    self.LABELS = open(labelsPath).read().strip().split("\n")

    weights_path = self.dir+"/yolov4-tiny.weights"
    config_path = self.dir+"/yolov4-tiny.cfg"

    print("PESOS",weights_path)

    self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    layer_indexes = self.model.getUnconnectedOutLayers()
    self.layer_names = [self.model.getLayerNames()[i - 1] for i in layer_indexes.flatten()]

    signal(SIGINT, self.handler)

    print(self.dir)

    print("Init Environment P3AT V2 (Image)")

  def handler(self, signal_received, frame):
    # Handle any cleanup here
    for i in range(10):
      self.stop_robot()

    self.failure=1
    #self.save_metrics("exp_05-real-10")
    print('SIGINT or CTRL-C detected. Stop robot!')
    time.sleep(1)
    exit(0)

  def callback(self, data):
    try:
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    results = self.pedestrian_detection(cv_image, self.model, self.layer_names,
            personidz=self.LABELS.index("person"))
    
    image_height, image_width = cv_image.shape[:2]
    center_x = image_width // 2
    center_y = image_height // 2
    rect_width = 60  # Largura do retângulo
    rect_height = 110  # Altura do retângulo
    rect_x = center_x - (rect_width // 2)
    rect_y = center_y - (rect_height // 2) - 10
    cv2.rectangle(cv_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (255, 0, 0), 2)

    for res in results:
    
      cv2.rectangle(cv_image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

      self.green_x1 = res[1][0] # Canto superior esquerdo
      self.green_y1 = res[1][1] # Canto superior esquerdo
      self.green_x2 = res[1][2] # Canto inferior direito
      self.green_y2 = res[1][3] # Canto inferior direito

      self.blue_x1 = rect_x # Canto superior esquerdo
      self.blue_y1 = rect_y # Canto superior esquerdo
      self.blue_x2 = rect_x + rect_width # Canto inferior direito
      self.blue_y2 = rect_y + rect_height # Canto inferior direito

      self.diff_left = self.green_x1 - self.blue_x1
      self.diff_right = self.blue_x2 - self.green_x2
      self.diff_top = self.green_y1 - self.blue_y1
      self.diff_bottom = self.blue_y2 - self.green_y2

    #x, y, largura, altura = self.width/3, 100, self.width/3, 250
    #cv2.rectangle(cv_image, (x, y), (x + largura, y + altura), (0, 0, 255), 2)
  
    if len(results) == 0: # Reset o ambiente em 5 ms
       #self.check_reset += 1
       #if self.check_reset > 10:
       time.sleep(1)
       self.reset()
  
    cv2.imshow("Detection",cv_image)

    key = cv2.waitKey(1)

    '''
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
    '''

  def pedestrian_detection(self, image, model, layer_names, personidz=0):
    (H, W) = image.shape[:2]
    results = []


    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_names)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > self.MIN_CONFIDENCE:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idzs = cv2.dnn.NMSBoxes(boxes, confidences, self.MIN_CONFIDENCE, self.NMS_THRESHOLD)
    # ensure at least one detection exists
    if len(idzs) > 0:
        # loop over the indexes we are keeping
        for i in idzs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
    # return the list of results
    return results

  def step(self, action):
    twist = Twist()
    twist.linear.x = self.actions[action][0]
    twist.angular.z = self.actions[action][1]

    self.max_steps -= 1

    execution_time = 0.2

    now = rospy.get_rostime()

    last_time = execution_time + now.secs

    while(last_time>now.secs):
      self.cmd_vel.publish(twist)
      now = rospy.get_rostime()

    ob=self.get_observation()
    lista = ob.tolist()

    if(self.verificar_faixa_valores(lista)):
      reward = 1
    else:
      reward = 0

    done = False

    if self.max_steps<=0:
      self.max_steps=20
      done = True
    
    return ob, reward, done, {}


  def distance(self, xA, xB, yA, yB):
    return(sqrt((xA-xB)**2) + ((yA-yB)**2))

  def stop_robot(self):
    twist = Twist()
    twist.linear.x = 0
    twist.angular.z = 0

    self.cmd_vel.publish(twist)

  def verificar_faixa_valores(self, lista, valor_min=-10, valor_max=10):
    for valor in lista:
      if valor < valor_min or valor > valor_max:
        return False
      if len(set(lista)) == 1:
        return False
    return True
  
  def get_observation(self):

    

    # Now send the request through the connection
    #result = self.sonar_service(self.sonar_response)

    #res = result.message.strip('][').split(', ')

    #result = list(map(lambda x: float(x.replace(",", "")), res))
    #print([self.diff_left, self.diff_right, self.diff_top, self.diff_bottom])
  
    return np.asarray([self.diff_left, self.diff_right, self.diff_top, self.diff_bottom])

  def reset(self):
    self.reset_world()

    if self.max_steps==0:
      print("Max Steps 20")

    #self.list_poses.clear()
    #self.check_reset = 0
    #time.sleep(0.2)
    obs=self.get_observation()

    return obs

  def render(self, mode='human', close=False):
    pass