cmake_minimum_required (VERSION 2.4.6)

include ($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set (ROS_BUILD_TYPE RelWithDebInfo)
rosbuild_init ()
rosbuild_add_boost_directories ()
add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fPIC)
rosbuild_check_for_sse ()

# Find OpenCV
find_package(OpenCV REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Dynamic reconfigure
rosbuild_find_ros_package(dynamic_reconfigure)
include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
gencfg()

#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3  -march=native ${SSE_FLAGS}" )

rosbuild_add_library (rgbd_evaluator
src/rgbd_evaluator.cpp
src/extract_rgbd_features.cpp
)

# Make nodelet
rosbuild_add_library (rgbd_evaluator_nodelet 
src/rgbd_evaluator_nodelet.cpp
)
target_link_libraries(rgbd_evaluator_nodelet ${OpenCV_LIBS})
target_link_libraries(rgbd_evaluator_nodelet rgbd_evaluator)

# Make node
rosbuild_add_executable (rgbd_evaluator_node 
src/rgbd_evaluator_node.cpp
)
target_link_libraries(rgbd_evaluator_nodelet ${OpenCV_LIBS})
target_link_libraries(rgbd_evaluator_node rgbd_evaluator)

# Make proprocessing app
rosbuild_add_executable (rgbd_preprocessing 
src/rgbd_evaluator_preprocessing.cpp
)
target_link_libraries(rgbd_evaluator_nodelet ${OpenCV_LIBS})
target_link_libraries(rgbd_preprocessing rgbd_evaluator)

