# Make shared library
add_library (opencv_ext
src/keypoint3d.cpp
)
target_link_libraries(opencv_ext ${OpenCV_LIBS})
