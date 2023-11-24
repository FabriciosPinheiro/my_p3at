#! /usr/bin/env python3
# remember to make this file executable (`chmod +x`) before trying to run it

import rospy
import message_filters
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import LaserScan, JointState, Range
from std_msgs.msg import Float64, Float32MultiArray
import time
import numpy as np

array_laser=[]

def discretize_observation(data,new_ranges):
	discretized_ranges = []
	min_range = 0.02
	done = False
	mod = len(data)/new_ranges
	for i, item in enumerate(data):
		if (i%mod==0):
			if data == float ('Inf'):
				discretized_ranges.append(6)
			elif np.isnan(data[i]):
				discretized_ranges.append(0)
			else:
				discretized_ranges.append(round(data[i],2))
		if (min_range > data[i] > 0):
			done = True

	return discretized_ranges

def discretize_obs(data,new_ranges):
	discretized_ranges = []

	index=np.linspace(0, len(data), new_ranges, dtype = np.int32)
	#print(index)
	x=0
	for i, item in enumerate(data):
		if i==index[x]:
			if data[i] == float ('Inf'):
				discretized_ranges.append(6)
			else:
				discretized_ranges.append(round(data[i],2))
			x+=1
		else:
			if (i+1)==len(data):
				if data[i] == float ('Inf'):
					discretized_ranges.append(6)
				else:
					discretized_ranges.append(round(data[i],2))
	
	return discretized_ranges

def callback(data):
	global array_laser
	array_laser = discretize_obs(data.ranges, 10)
	#print(array_laser)

def trigger_response(request):
	global array_laser
	return TriggerResponse(success=True, message=str(array_laser))

def laser_listener():
	rospy.init_node('Laser_Service', anonymous=True)

	rospy.Subscriber("/scan", LaserScan, callback)

	my_service = rospy.Service('/array_laser', Trigger, trigger_response)
	rospy.spin()

if __name__ == '__main__':
	laser_listener()