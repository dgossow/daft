cmake_minimum_required (VERSION 2.4.6)

include ($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

set (ROS_BUILD_TYPE RelWithDebInfo)
add_definitions (-Wall -O3 -Winvalid-pch -pipe -funroll-loops -fPIC)

#set (ROS_BUILD_TYPE Debug)

rosbuild_init ()
rosbuild_add_boost_directories ()
rosbuild_check_for_sse ()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Eigen
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories (${CMAKE_CURRENT_BINARY_DIR})
include_directories (${PROJECT_SOURCE_DIR})

# Dynamic reconfigure
#rosbuild_find_ros_package(dynamic_reconfigure)
#include(${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake)
#gencfg()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

# Find Eigen
#find_package(Eigen REQUIRED)
#include_directories(${EIGEN_INCLUDE_DIRS})
#add_definitions(${EIGEN_DEFINITIONS})

# Make OpenCV test node
rosbuild_add_executable (test_rgbd_cv
src/test_rgbd_cv_node.cpp
)
rosbuild_link_boost(test_rgbd_cv signals)
target_link_libraries(test_rgbd_cv ${OpenCV_LIBS})

# Make preprocessing app
rosbuild_add_executable (rgbd_preprocessing 
src/rgbd_preprocessing.cpp
)
target_link_libraries(rgbd_preprocessing  ${OpenCV_LIBS})

# Make extract_detector_file app
rosbuild_add_executable (extract_features 
src/extract_features.cpp
src/extract_keypoints.cpp
external/sift/Sift.cpp
external/parallelsurf/Image.cpp
external/parallelsurf/KeyPointDescriptor.cpp
external/parallelsurf/KeyPointDetector.cpp
external/parallelsurf/MathStuff.cpp  
)
target_link_libraries(extract_features ${OpenCV_LIBS})

# Make apply blur noise app
rosbuild_add_executable (apply_blur_noise 
src/apply_blur_noise.cpp
)
target_link_libraries(apply_blur_noise ${OpenCV_LIBS})

