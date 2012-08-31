cmake_minimum_required (VERSION 2.4.6)

project(daft)

# make daft findable in other catkin projects
find_package(catkin)

if(catkin_FOUND)
	message(STATUS "Found catkin.")
	catkin_stack()
	catkin_project(daft
	              DEPENDS Eigen Boost OpenCV
	              INCLUDE_DIRS opencv_features3d/include/ libdaft/include/
	              LIBRARIES daft_library
	)
endif(catkin_FOUND)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/opencv_features3d/include/)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/libdaft/include/)

find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})

add_subdirectory( opencv_features3d )
add_subdirectory( libdaft )
add_subdirectory( tools )

