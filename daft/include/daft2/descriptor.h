/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_DESCRIPTOR_H__
#define __DAFT2_DESCRIPTOR_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/features3d/features3d.hpp>

#include "filter_kernels.h"
#include "stuff.h"

#include <math.h>
#include <list>


//#define DESC_DEBUG_IMG

namespace cv
{
namespace daft2
{

Vec3f getNormal( const KeyPoint3D& kp, const cv::Mat1f depth_map, cv::Matx33f& K, float size_mult );

class SurfDescriptor
{
public:
  int getDescLen() { return 64; }

  bool getDesc( int patch_size, float thickness, Mat1f& smoothed_img, Mat1f& smoothed_img2,
      KeyPoint3D& kp, Mat1f& desc,
      const cv::Mat1f depth_map, cv::Matx33f& K, bool show_win=false );
};

}}

#endif
