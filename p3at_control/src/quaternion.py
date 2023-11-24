#!/usr/bin/env python3

import rospy
# tf.transformations alternative is not yet available in tf2
#from tf.transformations import quaternion_from_euler
from squaternion import Quaternion
  
if __name__ == '__main__':

  # RPY to convert: 90deg, 0, -90deg
  #q = quaternion_from_euler(0, 0, -1.57)

  #print("The quaternion representation is %s %s %s %s." % (q[0], q[1], q[2], q[3]))
  my_quaternion = Quaternion.from_euler(0, 0, 90, degrees=True)

  print(my_quaternion)