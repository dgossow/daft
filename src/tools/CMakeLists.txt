cmake_minimum_required (VERSION 2.4.6)

# use openmp (for parallel surf)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" )

# Find OpenCV
find_package(OpenCV REQUIRED)

include_directories (${CMAKE_CURRENT_SOURCE_DIR}/)

# Main app
add_executable (extract_daft_features 
extract_features.cpp
extract_impl.cpp
external/sift/Sift.cpp
external/parallelsurf/Image.cpp
external/parallelsurf/KeyPointDescriptor.cpp
external/parallelsurf/KeyPointDetector.cpp
external/parallelsurf/MathStuff.cpp  
)
target_link_libraries(extract_daft_features ${OpenCV_LIBS} daft)

