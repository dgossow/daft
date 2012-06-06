/*
 * feline.h
 *
 *  Created on: May 24, 2012
 *      Author: gossow
 */

#ifndef FELINE_H_
#define FELINE_H_

#include "gauss9x9.h"
#include "feline.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"

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
  return val / ( float(NumSteps) );//4)*minor_len*minor_len );
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


}
}
#endif /* FELINE_H_ */