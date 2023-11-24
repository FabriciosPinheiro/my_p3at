#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <sstream>
#include <math.h>
#include <iostream>
using namespace std;


double graus(double grau){
	return(grau*M_PI/180);
}


int main(int argc, char *argv[]){

	int param=0;
	std_msgs::Float64 msg;
	ros::init(argc, argv, "Controle_P3AT");
	ros::NodeHandle n("~");

	n.getParam("position", param);
  	//ROS_INFO("Got parameter : %d", param);

  	if(param==0)
  		param=1;
		
	ros::Publisher velocity_publisher = n.advertise<std_msgs::Float64>("/sim_p3at/joint1_position_controller/command",10);

	ros::Rate loop_rate(1);

	int count = 0;

	while(ros::ok() && count<3){
		msg.data = graus(param);

		//ROS_INFO("Value : %f", msg.data);	
		
		velocity_publisher.publish(msg);
					
		ros::spinOnce();

		loop_rate.sleep();
    	++count;
	}

	return 0;
}
