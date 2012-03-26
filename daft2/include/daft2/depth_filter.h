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

void smoothDepth( const Mat1f &scale_map,
    const Mat1d &ii_depth_map,
    const Mat_<uint64_t>& ii_depth_count,
    float base_scale,
    Mat1f &depth_out )
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  depth_out.create( ii_depth_map.rows-1, ii_depth_map.cols-1 );
  for ( int y = 0; y < ii_depth_map.rows-1; y++ )
  {
    for ( int x = 0; x < ii_depth_map.cols-1; ++x )
    {
      float s = scale_map[y][x] * base_scale + 0.5f;
      if ( s < 3 ) s=3;

      const int s_floor = s;
      const float t = s - s_floor;

      if ( !checkBounds( ii_depth_map, x, y, s_floor+1 ) )
      {
        depth_out(y,x) = nan;
        continue;
      }
      depth_out(y,x) = (1.0-t) * meanDepth( ii_depth_map, ii_depth_count, x, y, s_floor )
          + t * meanDepth( ii_depth_map, ii_depth_count, x, y, s_floor+1 );
    }
  }
}

}
}

#endif //rgbd_features_math_stuff_h_
