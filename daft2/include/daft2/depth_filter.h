/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_DEPTH_FILTER_H__
#define __DAFT2_DEPTH_FILTER_H__

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"
#include "kernel2d.h"

namespace cv
{
namespace daft2
{

void smoothDepth(
    const Mat1f &scale_map,
    const Mat1d &ii_depth_map,
    const Mat_<uint32_t>& ii_depth_count,
    float base_scale,
    Mat1f &depth_out );

void computeAffineMap(
    const Mat1f &scale_map,
    const Mat1f &depth_map,
    float sw,
    float min_px_scale,
    Mat4f& affine_map );

void computeAffineMapFixed(
    const Mat1f &depth_map,
    float sp,
    float f,
    Mat4f& affine_map );

}
}

#endif //rgbd_features_math_stuff_h_
