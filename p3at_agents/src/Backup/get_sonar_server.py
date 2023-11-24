#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import LaserScan, JointState, Range
from std_msgs.msg import Float64, Float32MultiArray
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import math
from itertools import cycle
import time

point_1, point_2, point_3 = 0.01, 1.57, -1.57
#q = quaternion_from_euler(0, 0, -1.57)
right_cam=True
pose_cam=0.0
value_sonar=0
array_sonar=[0,0,0]



def callback(sonar_value, cam_pose, cmd_vel):
    global pose_cam
    global value_sonar

    pose_cam=round(cam_pose.position[0], 2)

    value_sonar=round(sonar_value.range, 2)


def move_cam():
    global pose_cam
    global value_sonar
    global right_cam

    if(abs(point_1-pose_cam)<0.1) and (right_cam==False):
        cam_position.publish(point_2)
        array_sonar[0]=value_sonar

    if(abs(point_1-pose_cam)<0.1) and (right_cam==True):
        cam_position.publish(point_3)
        array_sonar[0]=value_sonar

    if(abs(point_2-pose_cam)<0.1):
        cam_position.publish(point_1)
        right_cam=True
        array_sonar[1]=value_sonar

    if(abs(point_3-pose_cam)<0.1):
        cam_position.publish(point_1)
        right_cam=False
        array_sonar[2]=value_sonar

    data_to_send = Float32MultiArray()  # the data to be sent, initialise the array
    data_to_send.data = array_sonar # assign the array with the value you want to send
    array_sonar_pub.publish(data_to_send)
   #print(array_sonar)


if __name__ == '__main__':
    rospy.init_node('pan_tilt')

    #cmd_vel = message_filters.Subscriber('/arduino/cmd_vel', Twist)
    cam_pose = message_filters.Subscriber('/arduino/joint_states', JointState)
    sonar_value = message_filters.Subscriber('/arduino/sonar', Range)

    ts = message_filters.ApproximateTimeSynchronizer([sonar_value, cam_pose], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    cam_position = rospy.Publisher('/arduino/joint1_position_controller/command', Float64, queue_size=10)
    array_sonar_pub = rospy.Publisher('/arduino/array_sonar', Float32MultiArray, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        move_cam()
    rate.sleep()