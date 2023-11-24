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

pose_cam, orientation_p3at, x_p3at, y_p3at=0.0,0.0,0.0,0.0
gama, k = 0.8, 1

def Teta_d(x, y):
    if(x==0.0):
        x = 0.1
    return(2.0*math.atan(y/x))

def NormRad(ang):
    while(ang>math.pi):
        ang = ang-2*math.pi
    while(ang<-math.pi):
        ang = ang+2*math.pi

    return(ang)

def SinC(theta):
    sinc = 1.0

    if (math.fabs(theta) > 0.001):
        return(math.sin(theta)/theta)
    else:
        return sinc

def Sign(x, y):
   if((x==0.0) and (y<0.0) or x>0.0):
        return 1.0
   else:
        return -1.0

def funcB1(x, y, theta, alpha):
    mybeta=0
    if(x!=0):
        mybeta = y/x
    theta_d = Teta_d(x,y)

    if(theta_d != 0):
        return math.cos(theta)*((theta_d/mybeta) - 1.0) + math.sin(theta)*(theta_d/2.0*(1.0-1.0/math.pow(mybeta,2.0)) + 1.0/mybeta)
    else:
        return math.cos(alpha)

def funcB2(x, y, theta):
    mybeta=0
    if(x!=0):
        mybeta = y/x
    theta_d = Teta_d(x,y)

    if (theta_d != 0):
        return math.cos(theta)*(2.0*mybeta/((1.0+math.pow(mybeta,2.0))*x)) - math.sin(theta)*(2.0/((1.0+math.pow(mybeta,2.0))*x))
    else:
        return math.sin(theta_d/2.0 - theta) * (2.0 / SinC(theta_d/2.0))

def graus(grau):
    return(grau*M_PI/180)

if __name__ == '__main__':

rospy.init_node('controller', anonymous=True)

p3at_pose = message_filters.Subscriber('/sim_p3at/pose', Odometry)

cam_pose = message_filters.Subscriber('/sim_p3at/joint_states', JointState)

ts = message_filters.ApproximateTimeSynchronizer([p3at_pose, cam_pose], 10, 0.1, allow_headerless=True)
ts.registerCallback(callback)

p3at_cmd_vel = rospy.Publisher('/sim_p3at/cmd_vel', Twist, queue_size=10)

cam_position = rospy.Publisher('/sim_p3at/joint1_position_controller/command', Float64, queue_size=10)

Ex, Ey, Eteta, xM, yM, a, alpha, beta, fb1, fb2 = 0,0,0,0,0,0,0,0,0,0
u, w, Erro_x, Erro_y, Erro_teta, teta_d = 0,0,0,0,0,0

Erro_teta = 0
Erro_x = 0
Erro_y = 0

rate = rospy.Rate(10) # 10hz
while not rospy.is_shutdown():
    
    Ex = x_p3at - Erro_x
    Ey = y_p3at - Erro_y
    Eteta = orientation_p3at - Erro_teta

    #ROS_INFO("Local -> Eixo_X: [%f],Eixo_Y: [%f], Angulo: [%f]", coord_x,coord_y,angulo);
    #rospy.loginfo("Local -> Eixo_X: [%f],Eixo_Y: [%f], Angulo: [%f]", x_p3at,y_p3at,orientation_p3at)

    xM = (math.cos(Erro_teta)*Ex + math.sin(Erro_teta)*Ey)
    yM = (-math.sin(Erro_teta)*Ex + math.cos(Erro_teta)*Ey)

    a = Sign(xM,yM)*math.sqrt(xM*xM + yM*yM)/(SinC(Teta_d(xM,yM)/2.0))
    alpha = NormRad(Eteta-Teta_d(xM,yM))

    fb1 = funcB1(xM,yM,Eteta, alpha)
    u = -gama/(1.0 + math.fabs(a))*fb1*a

    fb2 = funcB2(xM,yM,Eteta)
    w = (-fb2*u) - (k*alpha)
    

    #ROS_INFO("u, w: [%f, %f]",u,w);
    vel_msg = Twist()

   
    vel_msg.linear.x = u #seta a velocidade linear no eixo x
    
    vel_msg.angular.z = w

    p3at_cmd_vel.publish(vel_msg)
            
    rate.sleep()