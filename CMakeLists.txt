cmake_minimum_required (VERSION 2.4.6)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/opencv_ext/include/)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/libdaft/include/)

# Find OpenCV
# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories (${OpenCV_INCLUDE_DIR})

add_subdirectory( opencv_ext )

add_subdirectory( libdaft )
add_subdirectory( tools )

