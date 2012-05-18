/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_FILTER_KERNELS_H__
#define __DAFT2_FILTER_KERNELS_H__

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"

#include <eigen3/Eigen/Dense>

namespace cv
{
namespace daft2
{

template <int NumSteps>
inline float felineImpl( const Mat1d &ii, int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y,
    float half_axis_len )
{
  float start_x = major_x * half_axis_len;
  float start_y = major_y * half_axis_len;

  float val = 0;
  static const float t_norm = 2.0 / (NumSteps-1);

  for ( int step=0; step<NumSteps; step++ )
  {
    float t = float(step) * t_norm - 1.0;
    float x1 = (float)x - t*start_x;
    float y1 = (float)y - t*start_y;
    val += integrateBilinear( ii, x1-minor_len, y1-minor_len, x1+minor_len, y1+minor_len );
  }
  return val / ( float(NumSteps*4)*minor_len*minor_len );
}

inline float feline( const Mat1d &ii,
    int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y )
{
  if ( !checkBounds( ii, x, y, major_len+2 ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  const float half_axis_len = major_len-minor_len;
  return felineImpl<4>( ii, x, y, major_len, minor_len, major_x, major_y, half_axis_len );
  /*
  int num_steps = float(half_axis_len / (float)minor_len + 2.0f);

  if ( num_steps < 4 ) num_steps = 4;

  assert( num_steps >= 2 );

  //return float(num_steps) * 0.25;

  float start_x = major_x * half_axis_len;
  float start_y = major_y * half_axis_len;

  float val = 0;

  if ( num_steps == 2 )
  {
    for ( int step=0; step<2; step++ )
    {
      float t = float(step) * 2.0 - 1.0;
      float x1 = (float)x - t*start_x;
      float y1 = (float)y - t*start_y;
      val += integrateBilinear( ii, x1-minor_len, y1-minor_len, x1+minor_len, y1+minor_len );
    }
    val /= float( 8*minor_len*minor_len );
    return val;
  }
  else
  {
    for ( int step=0; step<num_steps; step++ )
    {
      float t = float(step)/float(num_steps-1) * 2.0 - 1.0;
      float x1 = (float)x - t*start_x + 0.5;
      float y1 = (float)y - t*start_y + 0.5;
      val += integrateBilinear( ii, x1-minor_len, y1-minor_len, x1+minor_len, y1+minor_len );
    }
    val /= float( num_steps * 4*minor_len*minor_len );
    return val;
  }
  */
}

/* Compute box mean with sub-integer scale interpolation */
inline float boxMean( const Mat1d &ii, int x, int y, float s )
{
  // make the box which cover the same area as a circle withradius s
  s *= 0.886f;
  if ( s < 1 ) s=1;
  const int si = int(s);
  const int si1 = si+1;
  if (!checkBounds( ii, x, y, si+1 ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float v1 = integrate ( ii, x - si,  y - si, x + si, y + si ) / float(4*si*si);
  const float v2 = integrate ( ii, x - si1,  y - si1, x + si1, y + si1 ) / float(4*si1*si1);
  return interpolateLinear(s - float(si), v1, v2);
}

inline float princCurvRatio( const Mat1f &response,
    int x, int y,
    float major_len,  float minor_len,
    float major_x1, float major_y1 )
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
