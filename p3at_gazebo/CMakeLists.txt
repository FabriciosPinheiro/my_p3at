cmake_minimum_required(VERSION 2.8.3)
project(p3at_gazebo)

## Add support for C++11
add_definitions(-std=c++11) # Adicionado

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  geometry_msgs
  nav_msgs
  gazebo_plugins
)

find_package(gazebo REQUIRED) # Adicionado

catkin_package(
INCLUDE_DIRS 
    include
  LIBRARIES 
     my_p3at
#    gazebo_ros_actor_command
    
  CATKIN_DEPENDS
    gazebo_ros
    gazebo_plugins
    geometry_msgs
    nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
  
include_directories(${GAZEBO_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
#src/my_p3at/p3at_gazebo/plugins/atorPlugin.cpp)

add_library(gazebo_ros_actor_plugin SHARED src/gazebo_ros_actor_command_plugin.cpp)
target_link_libraries(gazebo_ros_actor_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})

catkin_package(
  DEPENDS
    roscpp
    gazebo_ros
)
