cmake_minimum_required(VERSION 2.8.3)
project(p3at_control)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  nav_msgs 
  geometry_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

add_executable(command_cam src/cam_controller.cpp)
add_dependencies(command_cam nav_msgs_generate_messages_cpp)
target_link_libraries(command_cam ${catkin_LIBRARIES})

add_executable(teleop_gazebo src/teleop.cpp)
add_dependencies(teleop_gazebo nav_msgs_generate_messages_cpp)
target_link_libraries(teleop_gazebo ${catkin_LIBRARIES})

add_executable(teleop_p3at src/teleop_p3at.cpp)
add_dependencies(teleop_p3at nav_msgs_generate_messages_cpp)
target_link_libraries(teleop_p3at ${catkin_LIBRARIES})
