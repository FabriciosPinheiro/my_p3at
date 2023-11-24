#!/usr/bin/env python3

from __future__ import print_function

import rospy
from geometry_msgs.msg import Twist
from tkinter import *

class KeyboardEvent(Frame):
	def __init__(self):
		Frame.__init__(self)
		self.pack(expand=YES, fill=BOTH)
		self.master.title("Robot Teleop")
		self.master.geometry("300x100")

		#Values for max linear velocity and angular velocity
		self.max_value = 0.5

		# Label e Stringvar que vao exibir a tecla
		self.main_msg = StringVar()
		self.label = Label(self, textvariable = self.main_msg)
		self.main_msg.set("Use the arrow keys to move the robot \nand space for stop" )
		self.label.pack()

		self.msg = StringVar()
		self.label = Label(self, textvariable = self.msg)
		self.msg.set("waiting..." )
		self.label.pack()

		# Fazendo os binding no frame
		self.master.bind('<Left>', self.left_key)
		self.master.bind('<Right>', self.right_key)
		self.master.bind('<Up>', self.up_key)
		self.master.bind('<Down>', self.down_key)
		self.master.bind('<space>', self.Space)

		'''
		frame = Frame(self)
		bottomframe = Frame(self)
		bottomframe.pack( side = BOTTOM ) 
		redbutton = Button(bottomframe, text = 'Red', fg ='red') 
		redbutton.pack( side = LEFT) 
		greenbutton = Button(bottomframe, text = 'Brown', fg='brown') 
		greenbutton.pack( side = LEFT ) 
		bluebutton = Button(bottomframe, text ='Blue', fg ='blue') 
		bluebutton.pack( side = LEFT ) 
		blackbutton = Button(bottomframe, text ='Black', fg ='black') 
		blackbutton.pack( side = BOTTOM) 
		'''

		mainloop()
  
	def left_key(self, event):
		self.msg.set( "Left")
		twist = Twist()
		twist.linear.x = 0;
		twist.angular.z = self.max_value;
		pub.publish(twist)

	def right_key(self, event):
		self.msg.set( "Right")
		twist = Twist()
		twist.linear.x = 0;
		twist.angular.z = -self.max_value;
		pub.publish(twist)

	def up_key(self, event):
		self.msg.set( "Up")
		twist = Twist()
		twist.linear.x = self.max_value;
		twist.angular.z = 0;
		pub.publish(twist)

	def down_key(self, event):
		self.msg.set( "Down")
		twist = Twist()
		twist.linear.x = -self.max_value;
		twist.angular.z = 0;
		pub.publish(twist)

	def Space(self, event):
		self.msg.set( "Stop")
		twist = Twist()
		twist.linear.x = 0;
		twist.angular.z = 0;
		pub.publish(twist)

if __name__ == "__main__":

	rospy.init_node('teleop_twist_keyboard')
	pub = rospy.Publisher('sim_p3at/cmd_vel', Twist, queue_size = 1)	
	KeyboardEvent()