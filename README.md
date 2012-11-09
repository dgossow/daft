DAFT
====

Implemenation of DAFT (Depth-Adaptive Feature Transform) algorithm, plus tools. 
For more information, see https://ias.in.tum.de/people/gossow/rgbd.

# Directory structure:

*  **eval:**    Matlab evaluation framework
*  **libdaft:** DAFT implementation
*  **opencv_ext:** OpenCV addons (contains Keypoint3D class)
*  **test_images:** printable images for debugging purposes
*  **tools:**       command-line tools for feature extraction

# Checkout & compile:


The following instructions have been tested with Ubuntu 11.10
and OpenCV 2.3.

You can install OpenCV like this:

    sudo apt-get install libopencv2.3

For an out-of-source build, do:

    git clone https://github.com/dgossow/daft.git
    mkdir daft_build
    cd daft_build
    cmake ../daft
    make
    
This will create the static library *libdaft/libdaft.a*.
You will need to link against it and also have 
your include paths set up.

Extract DAFT features from an image like so:

    #include <daft/daft.h>
    
    // ...

    cv::Mat gray_img;
    cv::Mat mask_img;
    cv::Mat depth_img;
    cv::Matx33f K;
    
    // load data ..
    
    std::vector<cv::KeyPoint3D> keypoints;
    cv::Mat descriptors;
    
    cv::daft::DAFT daft;
    daft( gray_img, mask_img, depth_img, K, keypoints, descriptors );
  

