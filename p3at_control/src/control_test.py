#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import LaserScan, JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import math
from itertools import cycle
import time

point_1, point_2, point_3 = 0.01, 1.57, -1.57
positions = [0.01, 1.57, -1.57]
control = 0
#q = quaternion_from_euler(0, 0, -1.57)
right_cam=True
right_p3at=True
pose_cam, orientation_p3at, x_p3at, y_p3at=0.0,0.0,0.0,0.0


def callback(p3at_pose, cam_pose):
    global pose_cam
    global orientation_p3at, x_p3at, y_p3at
    pose_cam=round(cam_pose.position[0], 2)
    #orientation_p3at=round(p3at_pose.pose.pose.orientation.z, 2)
    #x_p3at=p3at_pose.pose.pose.position.x
    #y_p3at=p3at_pose.pose.pose.position.y

    rot_q = p3at_pose.pose.pose.orientation
    (x_p3at, y_p3at, orientation_p3at) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
    #print(position)
    
    #print("Orientation: %.2f" % (p3at_pose.pose.pose.orientation.z))
    #print("Position cam: %.2f" % (cam_pose.position[0]))

def move_cam():
    global pose_cam
    global right_cam

    #print(right)
    #print("point_1: ",(abs(point_1-position)))
    #print("point_2: ",(abs(point_2-position)))
    #print("point_3: ",(abs(point_3-position)))

    if(abs(point_1-pose_cam)<0.05) and (right_cam==False):
        cam_position.publish(point_2)

    if(abs(point_1-pose_cam)<0.05) and (right_cam==True):
        cam_position.publish(point_3)

    if(abs(point_2-pose_cam)<0.05):
        cam_position.publish(point_1)
        right_cam=True

    if(abs(point_3-pose_cam)<0.05):
        cam_position.publish(point_1)
        right_cam=False
 

def imit_cam():
    global orientation_p3at, x_p3at, y_p3at, k, q, control

    vel_msg = Twist()

    vel_msg.linear.x = 0
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    #rospy.loginfo("orientation: [%f], Q2: [%f], result: [%f]", orientation_p3at, q1, abs(q1-orientation_p3at))
    
    
    if(control==0):
        if(abs(positions[2]-orientation_p3at)>0.1):
            #rospy.loginfo("orientation: [%f], i: [%f], result: [%f]", orientation_p3at, positions[2], abs(positions[2]-orientation_p3at))
            vel_msg.angular.z = -0.5
            p3at_cmd_vel.publish(vel_msg)
        else:
            vel_msg.angular.z = 0.0
            p3at_cmd_vel.publish(vel_msg)
            control=1
            time.sleep(2)

    if(control==1):
        if(abs(positions[0]-orientation_p3at)>0.1):
            #rospy.loginfo("orientation: [%f], i: [%f], result: [%f]", orientation_p3at, positions[0], abs(positions[0]-orientation_p3at))
            vel_msg.angular.z = 0.5
            p3at_cmd_vel.publish(vel_msg)
        else:
            vel_msg.angular.z = 0.0
            p3at_cmd_vel.publish(vel_msg)
            control=2
            time.sleep(2)

    if(control==2):
        if(abs(positions[1]-orientation_p3at)>0.1):
            #rospy.loginfo("orientation: [%f], i: [%f], result: [%f]", orientation_p3at, positions[1], abs(positions[1]-orientation_p3at))
            vel_msg.angular.z = 0.5
            p3at_cmd_vel.publish(vel_msg)
        else:
            vel_msg.angular.z = 0.0
            p3at_cmd_vel.publish(vel_msg)
            control=3
            time.sleep(2)

    if(control==3):
        if(abs(positions[0]-orientation_p3at)>0.1):
            #rospy.loginfo("orientation: [%f], i: [%f], result: [%f]", orientation_p3at, positions[0], abs(positions[0]-orientation_p3at))
            vel_msg.angular.z = -0.5
            p3at_cmd_vel.publish(vel_msg)
        else:
            vel_msg.angular.z = 0.0
            p3at_cmd_vel.publish(vel_msg)
            control=0
            time.sleep(2)

    #rospy.loginfo("control: [%d]", control)


if __name__ == '__main__':

    rospy.init_node('controller', anonymous=True)

    p3at_pose = message_filters.Subscriber('/sim_p3at/pose', Odometry)

    cam_pose = message_filters.Subscriber('/sim_p3at/joint_states', JointState)

    ts = message_filters.ApproximateTimeSynchronizer([p3at_pose, cam_pose], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    
    p3at_cmd_vel = rospy.Publisher('/sim_p3at/cmd_vel', Twist, queue_size=10)

    cam_position = rospy.Publisher('/sim_p3at/joint1_position_controller/command', Float64, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        #imit_cam()
        move_cam()
    rate.sleep()