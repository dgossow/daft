cmake_minimum_required (VERSION 2.4.6)

project(daft)

# 
if(catkin_FOUND)
	message(STATUS "Found catkin.")
	catkin_stack()
	catkin_project(daft
		DEPENDS Eigen Boost OpenCV
		INCLUDE_DIRS include/
		LIBRARIES daft
	)
else(catkin_FOUND)
	set( CMAKE_BUILD_TYPE RelWithDebInfo)
endif(catkin_FOUND)


# make daft findable in other catkin projects
find_package(catkin)

# Setup compiler optimizations etc.
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})

# make sure all projects find the headers 
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include/)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include/daft/)

add_subdirectory( src/daft )
add_subdirectory( src/tools )
add_subdirectory( test )

