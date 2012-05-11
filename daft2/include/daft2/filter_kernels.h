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
#include "kernel2d.h"

namespace cv
{
namespace daft2
{

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
  const int num_steps = float(half_axis_len / (float)minor_len + 2.0f);

  assert( num_steps >= 2 );

  //return float(num_steps) * 0.25;

  const int minor_len_floor = minor_len;
  const int minor_len_ceil = minor_len_floor + 1;

  float start_x = major_x * half_axis_len;
  float start_y = major_y * half_axis_len;

  float val_floor = 0;
  float val_ceil = 0;

  if ( num_steps == 2 )
  {
    for ( int step=0; step<2; step++ )
    {
      float t = float(step) * 2.0 - 1.0;
      int x1 = (float)x - t*start_x + 0.5;
      int y1 = (float)y - t*start_y + 0.5;
      val_floor += integrate( ii, x1-minor_len_floor, y1-minor_len_floor, x1+minor_len_floor, y1+minor_len_floor );
      val_ceil += integrate( ii, x1-minor_len_ceil, y1-minor_len_ceil, x1+minor_len_ceil, y1+minor_len_ceil );
    }
    val_floor /= float( 8*minor_len_floor*minor_len_floor );
    val_ceil /= float( 8*minor_len_ceil*minor_len_ceil );
    return interpolateLinear( minor_len - minor_len_floor, val_floor, val_ceil );
  }
  else
  {
    for ( int step=0; step<num_steps; step++ )
    {
      float t = float(step)/float(num_steps-1) * 2.0 - 1.0;
      int x1 = (float)x - t*start_x + 0.5;
      int y1 = (float)y - t*start_y + 0.5;
      val_floor += integrate( ii, x1-minor_len_floor, y1-minor_len_floor, x1+minor_len_floor, y1+minor_len_floor );
      val_ceil += integrate( ii, x1-minor_len_ceil, y1-minor_len_ceil, x1+minor_len_ceil, y1+minor_len_ceil );
    }
    val_floor /= float( num_steps * 4*minor_len_floor*minor_len_floor );
    val_ceil /= float( num_steps * 4*minor_len_ceil*minor_len_ceil );
    return interpolateLinear( minor_len - minor_len_floor, val_floor, val_ceil );
  }
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

inline float princCurvRatioImpl( const Mat1d &ii, int x, int y, int a )
{
  if (!checkBounds( ii, x, y, 6*a ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  float values[9][9];
  integrateGridCentered<double,9>(ii, x, y, a, (float*)values);

  float dxx = sDxxKernel.convolve(values);
  float dyy = sDyyKernel.convolve(values);
  float dxy = sDxyKernel.convolve(values);

  float trace = dxx + dyy;
  float det = dxx*dyy - (dxy*dxy);

  if ( det <= 0 )
  {
    return std::numeric_limits<float>::max();
  }

  return trace*trace/det;
}

inline float princCurvRatio( const Mat1d &ii, int x, int y, float s )
{
  float a = 0.5893f * s; // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float v1 = princCurvRatioImpl(ii, x, y, ai);
  float v2 = princCurvRatioImpl(ii, x, y, ai + 1);
  return interpolateLinear(t, v1, v2);
}

inline float princCurvRatioAffineImpl( const Mat1d &ii, int x, int y, int a, float ratio, float angle )
{
  if (!checkBounds( ii, x, y, 6*a ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  float values[9][9];
  integrateGridCentered<double,9>(ii, x, y, a, (float*)values);

  float dxx = sDxxKernelCache.convolve(values, ratio, angle);
  float dyy = sDyyKernelCache.convolve(values, ratio, angle);
  float dxy = sDxyKernelCache.convolve(values, ratio, angle);

  float trace = dxx + dyy;
  float det = dxx*dyy - (dxy*dxy);

  if ( det <= 0 )
  {
    return std::numeric_limits<float>::max();
  }

  return trace*trace/det;
}

inline float princCurvRatioAffine( const Mat1d &ii, int x, int y, float major, float minor, float angle )
{
  float a = 0.5893f * major; // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float ratio = minor / major;
  float v1 = princCurvRatioAffineImpl(ii, x, y, ai, ratio, angle);
  float v2 = princCurvRatioAffineImpl(ii, x, y, ai + 1, ratio, angle);
  return interpolateLinear(t, v1, v2);
}

}
}

#endif //rgbd_features_math_stuff_h_
