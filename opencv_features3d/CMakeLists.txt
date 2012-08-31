# Make shared library
add_library (opencv_features3d
src/keypoint3d.cpp
)
target_link_libraries(opencv_features3d ${OpenCV_LIBS})
