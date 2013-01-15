cmake_minimum_required (VERSION 2.4.6)

set ( CMAKE_BUILD_TYPE RelWithDebInfo)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -funroll-loops -O3  -march=native ${SSE_FLAGS} -DNDEBUG" )

#set (CMAKE_BUILD_TYPE Debug)

#message

# Make shared library
add_library (daft
daft.cpp
feature_detection.cpp
descriptor.cpp
depth_filter.cpp
keypoint3d.cpp
)
target_link_libraries(daft ${OpenCV_LIBS})

