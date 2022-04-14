cmake_minimum_required(VERSION 2.8.3)
project(UrbanFlyCEMPlanner)

#set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake-modules") 

set(dynamicEDT3D_DIR "/home/sudarshan/octomap/lib/cmake/dynamicEDT3D/")
set(DYNAMICEDT3D_LIBRARIES "/home/sudarshan/octomap/dynamicEDT3D/lib/libdynamicedt3d.so")

set(AIRSIM_ROOT "/home/sudarshan/Planner_ws/src/AirSim" )


find_package(catkin REQUIRED COMPONENTS
    pcl_conversions
    pcl_ros
    roscpp
    rospy
    std_msgs
    geometry_msgs
    message_filters
    nav_msgs
    sensor_msgs
    tf
    cv_bridge
    )

include_directories(
  ${AIRSIM_ROOT}/HelloDrone
  ${AIRSIM_ROOT}/AirLib/include
  ${RPC_LIB_INCLUDES}
  ${AIRSIM_ROOT}/MavLinkCom/include
  ${AIRSIM_ROOT}/MavLinkCom/common_utils
)


find_package(OpenCV REQUIRED)
find_package(dynamicEDT3D )
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")


find_package(Ceres REQUIRED)
include_directories( ${DYNAMICEDT3D_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

#add_executable(rpvio_mapper
#    src/parameters.cpp
#    src/VPDetection.cpp
#    src/mapper_node.cpp
#)

#add_executable(rpvio_planner
#    src/planner_node.cpp
#    src/parameters.cpp
#)

add_executable(CEMPlanner
    src/PlannerVins.cpp
    src/kinodynamic_astar_Planar.cpp
    src/parameters.cpp

)

#add_executable(rpvio_estimator
#    src/estimator_node.cpp
#    src/parameters.cpp
#    src/estimator.cpp
#    src/feature_manager.cpp
#    src/factor/pose_local_parameterization.cpp
#    src/factor/projection_factor.cpp
#    src/factor/projection_td_factor.cpp
#    src/factor/marginalization_factor.cpp
#    src/utility/utility.cpp
#    src/utility/visualization.cpp
#    src/utility/CameraPoseVisualization.cpp
#   src/initial/solve_5pts.cpp
#    src/initial/initial_aligment.cpp
#    src/initial/initial_sfm.cpp
#    src/initial/initial_ex_rotation.cpp
#    )




#target_link_libraries(rpvio_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 
#target_link_libraries(rpvio_mapper ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES}) 
#target_link_libraries(rpvio_planner ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 
target_link_libraries(CEMPlanner ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${DYNAMICEDT3D_LIBRARIES}   ) 
#<!-- target_link_libraries(pcNoise ${catkin_LIBRARIES}) -->

#catkin_install_python(PROGRAMS 
#  scripts/planner_node.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)


