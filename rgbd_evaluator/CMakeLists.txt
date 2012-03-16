cmake_minimum_required (VERSION 2.4.6)

include ($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set (ROS_BUILD_TYPE RelWithDebInfo)
add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fPIC)

#set (ROS_BUILD_TYPE Debug)

rosbuild_init ()
#rosbuild_add_boost_directories ()
rosbuild_check_for_sse ()

# Find OpenCV
find_package(OpenCV REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories (${CMAKE_CURRENT_BINARY_DIR})

# Dynamic reconfigure
rosbuild_find_ros_package(dynamic_reconfigure)
include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
gencfg()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

# Make OpenCV test node
rosbuild_add_executable (test_rgbd_cv
src/test_rgbd_cv_node.cpp
)
# Make OpenCV test node
rosbuild_add_executable (test_transpose
src/test_transpose_node.cpp
)

# Make preprocessing app
rosbuild_add_executable (rgbd_preprocessing 
src/rgbd_evaluator_preprocessing.cpp
)
#target_link_libraries(rgbd_evaluator_nodelet ${OpenCV_LIBS})
#target_link_libraries(rgbd_preprocessing rgbd_evaluator)

# Make extract_detector_file app
rosbuild_add_executable (extract_detector 
src/extract_detector_file.cpp
)
#target_link_libraries(rgbd_evaluator_nodelet ${OpenCV_LIBS})
#target_link_libraries(extract_detector rgbd_evaluator)

# Make apply blur noise app
rosbuild_add_executable (apply_blur_noise 
src/apply_blur_noise.cpp
)