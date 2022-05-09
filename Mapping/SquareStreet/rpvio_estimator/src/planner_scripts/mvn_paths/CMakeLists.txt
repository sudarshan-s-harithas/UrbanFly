cmake_minimum_required (VERSION 3.0)
set (CMAKE_CXX_STANDARD 11)
project (mvn_paths)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
 
add_executable (example example.cpp)
target_link_libraries (example Eigen3::Eigen)

