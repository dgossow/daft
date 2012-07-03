cmake_minimum_required (VERSION 2.4.6)

set (EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set (LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/opencv_ext/include/)
include_directories (${CMAKE_CURRENT_SOURCE_DIR}/libdaft/include/)

add_subdirectory( opencv_ext )
add_subdirectory( libdaft )
add_subdirectory( tools )

