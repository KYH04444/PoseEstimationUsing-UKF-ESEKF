cmake_minimum_required(VERSION 2.8.3)
project(uwb_imu)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  roscpp
  rospy
  std_msgs
  rviz
  tf
  urdf
  xacro
  sensor_msgs
  message_generation
)

find_package(Eigen3 REQUIRED)

add_message_files(
  FILES
  UwbMsg.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs 
)
catkin_package(
  LIBRARIES uwb_imu uwb_imu_ukf
   CATKIN_DEPENDS geometry_msgs mavros_msgs roscpp rospy std_msgs message_runtime

)

# catkin_package(
#   LIBRARIES uwb_imu_ukf 
#    CATKIN_DEPENDS geometry_msgs mavros_msgs roscpp rospy std_msgs message_runtime

# )
# include_directories("/usr/include/eigen3")
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_install_python(PROGRAMS scripts/traj_plot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(uwb_imu src/imu_uwb_fusion.cpp)
add_executable(imu_uwb_fusion_node src/imu_uwb_fusion_node.cpp)
target_link_libraries(imu_uwb_fusion_node ${catkin_LIBRARIES} uwb_imu)

add_library(uwb_imu_ukf src/imu_uwb_fusion_ukf.cpp)
add_executable(imu_uwb_fusion_ukf_node src/imu_uwb_fusion_ukf_node.cpp)
target_link_libraries(imu_uwb_fusion_ukf_node ${catkin_LIBRARIES} uwb_imu_ukf)

add_executable(uwb_filtered src/uwb_filtered.cpp)
target_link_libraries(uwb_filtered ${catkin_LIBRARIES} )

add_executable(moving_average_filter_node src/moving_average_filter.cpp)
target_link_libraries(moving_average_filter_node ${catkin_LIBRARIES})

# add_library(uwb_imu src/imu_uwb_fusion.cpp)
# target_link_libraries(uwb_imu ${catkin_LIBRARIES})

# add_executable(imu_uwb_fusion_node src/imu_uwb_fusion_node.cpp)
# target_link_libraries(imu_uwb_fusion_node uwb_imu)

# add_library(uwb_imu_ukf src/imu_uwb_fusion_ukf.cpp)
# target_link_libraries(uwb_imu_ukf ${catkin_LIBRARIES})

# add_executable(imu_uwb_fusion_ukf_node src/imu_uwb_fusion_ukf_node.cpp)
# target_link_libraries(imu_uwb_fusion_ukf_node uwb_imu_ukf)

# add_executable(uwb_filtered src/uwb_filtered.cpp)
# target_link_libraries(uwb_filtered ${catkin_LIBRARIES})

# add_executable(moving_average_filter_node src/moving_average_filter.cpp)
# target_link_libraries(moving_average_filter_node ${catkin_LIBRARIES})
