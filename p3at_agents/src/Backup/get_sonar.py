#!/usr/bin/env python

import sys
import rospy
from arduino_agets.srv import *

if __name__ == "__main__":
    rospy.wait_for_service('add_two_ints')
    try:
        value = rospy.ServiceProxy('add_two_ints', SonarValue)
        print(value)
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e