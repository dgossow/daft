cmake_minimum_required (VERSION 2.4.6)

set ( CMAKE_BUILD_TYPE RelWithDebInfo)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

#set (CMAKE_BUILD_TYPE Debug)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/include/daft/)

# Make shared library
add_library (daft
src/daft.cpp
src/feature_detection.cpp
src/descriptor.cpp
src/depth_filter.cpp
)
target_link_libraries(daft ${OpenCV_LIBS} opencv_ext)

add_executable (test1 
test/test1.cpp
)
target_link_libraries(test1 daft)

