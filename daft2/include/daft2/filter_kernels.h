/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_FILTER_KERNELS_H__
#define __DAFT2_FILTER_KERNELS_H__

#include <opencv2/opencv.hpp>

#include "feline.h"
#include "stuff.h"


namespace cv
{
namespace daft2
{

inline float princCurvRatio( const Mat1f &response,
    int x, int y,
    float major_len,  float minor_len,
    float major_x1, float major_y1 )
    //Mat& disp )
{
  const float major_x = major_len * major_x1;
  const float major_y = major_len * major_y1;
  const float minor_x = minor_len * major_y1;
  const float minor_y = minor_len * -major_x1;

  float values[3][3];

  for( int u=-1;u<=1;u++ )
  {
    for( int v=-1;v<=1;v++ )
    {
      int x1 = x + u*major_x + v*minor_x;
      int y1 = y + u*major_y + v*minor_y;
      if (!checkBounds( response, x1, y1, 1 ))
      {
        return std::numeric_limits<float>::max();
      }
      values[v+1][u+1] = response( y1, x1 );

      //cv::circle( disp, Point(x1,y1 ), 2.0, Scalar(255,255,255), 1 );
    }
  }

#if 1
  float dxx = values[1][2] + values[1][0] - 2*values[1][1];
  float dyy = values[2][1] + values[0][1] - 2*values[1][1];
  float dxy = values[2][2] + values[0][0] - values[2][0] - values[0][2];
#else
  float dxx = values[0][0] + values[1][0] + values[2][0] \
      + values[0][2] + values[1][2] + values[2][2] \
      - 2*values[1][0] - 2*values[1][1] - 2*values[1][2];

  float dyy = values[0][0] + values[0][1] + values[0][2] \
      + values[2][0] + values[2][1] + values[2][2] \
      - 2*values[0][1] - 2*values[1][1] - 2*values[2][1];

  float dxy = values[2][2] + values[0][0] - values[2][0] - values[0][2];

  dxx /= 6.0;
  dyy /= 6.0;
  dxy /= 2.0;
#endif

  float trace = dxx + dyy;
  float det = dxx*dyy - (dxy*dxy);

  float r_val = trace*trace/det;

  return r_val;
}


}
}

#endif //rgbd_features_math_stuff_h_
