/*
 * feline.h
 *
 *  Created on: May 24, 2012
 *      Author: gossow
 */

#ifndef FELINE_H_
#define FELINE_H_

#include "feline.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"

namespace cv
{
namespace daft2
{

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
  return interpolateLinear<float,float>(s - float(si), v1, v2);
}


template <int NumSteps>
inline float felineImpl( const Mat1d &ii, int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y,
    float half_axis_len )
{
  float start_x = major_x * half_axis_len;
  float start_y = major_y * half_axis_len;

  float val = 0;
  float area = 0;

  static const float t_norm = 2.0 / (NumSteps-1);

  for ( int step=0; step<NumSteps; step++ )
  {
    float t = float(step) * t_norm - 1.0;
    float x1 = (float)x - t*start_x + 0.5;
    float y1 = (float)y - t*start_y + 0.5;

    const int x_low = x1-minor_len;
    const int y_low = y1-minor_len;
    const int x_high = x1+minor_len;
    const int y_high = y1+minor_len;

    val += integrate( ii, x_low, y_low, x_high, y_high );
    area += (x_high-x_low)*(y_high-y_low);
  }
  return val / area;
}

inline float feline( const Mat1d &ii,
    int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y )
{
  assert(minor_len <= major_len);

  if ( !checkBounds( ii, x, y, major_len+4 ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  float f_probes = 2.0f * (major_len/minor_len) - 1.0;
  int i_probes = f_probes + 0.5f;

  const float half_axis_len = major_len-minor_len;

  switch( i_probes )
  {
  case 1:
    return boxMean( ii, x, y, major_len );
  case 2:
    return felineImpl<2>( ii, x, y, major_len, minor_len, major_x, major_y, half_axis_len );
  case 3:
    return felineImpl<3>( ii, x, y, major_len, minor_len, major_x, major_y, half_axis_len );
  case 4:
    return felineImpl<4>( ii, x, y, major_len, minor_len, major_x, major_y, half_axis_len );
  default:
    return felineImpl<5>( ii, x, y, major_len, minor_len, major_x, major_y, half_axis_len );
  }

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


}
}
#endif /* FELINE_H_ */
