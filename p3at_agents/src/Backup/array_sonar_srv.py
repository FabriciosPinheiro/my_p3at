#! /usr/bin/env python
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import message_filters
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import LaserScan, JointState, Range
from std_msgs.msg import Float64, Float32MultiArray
import time

point_1, point_2, point_3 = 0.01, 1.57, -1.57
#q = quaternion_from_euler(0, 0, -1.57)
pose_cam=0.0
value_sonar=0
array_sonar=[0,0,0]

def callback(sonar_value, cam_pose):
	global pose_cam
	global value_sonar

	pose_cam=round(cam_pose.position[0], 2)

	value_sonar=round(sonar_value.range, 2)

def trigger_response(request):
	global pose_cam
	global value_sonar

	move_cam()

	return TriggerResponse(success=True, message=str(array_sonar))

def move_cam():
	global pose_cam
	global value_sonar
	global array_sonar

	array_sonar=[0,0,0]

	# Front
	cam_position.publish(point_1)
	time.sleep(1)
	array_sonar[0]=value_sonar

	# left
	cam_position.publish(point_2)
	time.sleep(1)
	array_sonar[1]=value_sonar

	# right
	cam_position.publish(point_3)
	time.sleep(1)
	array_sonar[2]=value_sonar

	#time.sleep(1)
	# Origin position
	cam_position.publish(point_1)


rospy.init_node('Sonar_Service')
cam_pose = message_filters.Subscriber('/arduino/joint_states', JointState)
sonar_value = message_filters.Subscriber('/arduino/sonar', Range)

cam_position = rospy.Publisher('/arduino/joint1_position_controller/command', Float64, queue_size=10)

ts = message_filters.ApproximateTimeSynchronizer([sonar_value, cam_pose], 10, 0.1, allow_headerless=False)
ts.registerCallback(callback)

my_service = rospy.Service('/array_sonar', Trigger, trigger_response)
rospy.spin()