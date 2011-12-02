cmake_minimum_required (VERSION 2.4.6)

include ($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set (ROS_BUILD_TYPE RelWithDebInfo)
rosbuild_init ()
#rosbuild_add_boost_directories ()
add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fPIC)
rosbuild_check_for_sse ()

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Dynamic reconfigure
rosbuild_find_ros_package(dynamic_reconfigure)
include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
gencfg()

# Make .so from ExtractFeatures class
#${SSE_FLAGS}
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} -O3  -march=native -fopenmp ${SSE_FLAGS}" )
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3  -march=native -fopenmp -fPIC ${SSE_FLAGS}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3  -march=native ${SSE_FLAGS}" )

# Make node
rosbuild_add_executable (rgbd_evaluator_node 
src/rgbd_evaluator.cpp
src/extract_rgbd_features.cpp
src/rgbd_evaluator_node.cpp
)

