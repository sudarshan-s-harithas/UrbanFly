cmake_minimum_required(VERSION 2.8.3)
project(rpvio_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

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

find_package(OpenCV REQUIRED)

add_service_files(
        FILES
        PlaneSegmentation.srv
)

generate_messages(
        DEPENDENCIES
        sensor_msgs
        std_msgs
)


# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)


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

add_executable(rpvio_mapper
    src/parameters.cpp
    src/VPDetection.cpp
    src/local_map.cpp
    src/mapper_node.cpp
        )

add_executable(rpvio_planner
    src/planner_node.cpp
    src/parameters.cpp
)

add_executable(rpvio_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )

add_executable(rpvio_preprocessor src/preprocessor_node.cpp src/OpticalFlowPropagator.cpp)
target_link_libraries(rpvio_preprocessor ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
add_dependencies(rpvio_preprocessor rpvio_estimator_gencpp)


target_link_libraries(rpvio_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(rpvio_mapper ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(rpvio_planner ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})

#catkin_install_python(PROGRAMS
#  scripts/planner_node.py
#        scripts/planercnn/plane_segmentation_service_server.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)


