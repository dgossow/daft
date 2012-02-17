cmake_minimum_required (VERSION 2.4.6)

include ($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set (ROS_BUILD_TYPE RelWithDebInfo)
#set (ROS_BUILD_TYPE Debug)

rosbuild_init ()
rosbuild_add_boost_directories ()
#add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fPIC)
rosbuild_check_for_sse ()

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (${PROJECT_SOURCE_DIR}/include/rgbd_features_cv/)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

# Make shared library
rosbuild_add_library (rgbd_features_cv
src/daft.cpp
src/feature_detection.cpp
src/keypoint3d.cpp
)

target_link_libraries(rgbd_features_cv ${OpenCV_LIBS})
