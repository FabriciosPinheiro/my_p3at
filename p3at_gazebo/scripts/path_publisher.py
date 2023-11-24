#!/usr/bin/env python3
import math
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler

import random

def publish_path():
    # Initialize the node
    rospy.init_node('path_publisher_node')
    
    # Create the publisher with topic "/cmd_path" and message type "Path"
    pub = rospy.Publisher('/cmd_path', Path, queue_size=1, latch=True)
    
    # Create the Path message
    path_msg = Path()
    
    # Set the header for the Path message
    path_msg.header.stamp = rospy.Time.now()
    path_msg.header.frame_id = "map"
    
    # Create the PoseStamped messages for each waypoint
    num_waypoints = 50  # Number of waypoints on the path
    #waypontsQuant = 0
    for i in range(num_waypoints):
        pose = PoseStamped()

        #pose.header.waypontsQuant += i

        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        #angle = i * (2 * 3.14159 / num_waypoints)
        #pose.pose.position = Point(x=2 * math.cos(angle), y=2 * math.sin(angle), z=0)

        #Generate random coordinates
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = 0

        pose.pose.position = Point(x=x, y=y, z=z)

        # Calculate the tangent vector at the current point

        # Set the orientation of the PoseStamped message to the quaternion that represents the rotation from the positive x-axis to the tangent vector
        #quat = Quaternion(*quaternion_from_euler(0, 0, angle+(3.14159/2.0)))
        quat = Quaternion(*quaternion_from_euler(0, 0, random.uniform(0, 2*math.pi)))
        pose.pose.orientation = quat
        path_msg.poses.append(pose)
    
    # Publish the Path message to the "/cmd_path" topic
    rate = rospy.Rate(10)  # Publish at 10 Hz
    pub.publish(path_msg)
    print ('published path', path_msg)
    while not rospy.is_shutdown():
        # pub.publish(path_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_path()
    except rospy.ROSInterruptException:
        pass
