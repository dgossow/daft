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
#include "interpolation.h"

namespace cv
{
namespace daft
{

/* Compute box mean with sub-integer scale interpolation */
inline float boxMean( const Mat1d &ii, int x, int y, float s )
{
  // make the box cover the same area as a circle with radius s
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
  return interp< float, float, inter::linear<float> >(s - float(si), v1, v2);
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

  const float t_norm = 2.0 / (NumSteps-1);

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

  if ( !checkBounds( ii, x, y, major_len+20 ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }

  //original feline impl:
  float f_probes = 2.0f * (major_len/minor_len) - 1.0;
  int i_probes = f_probes + 0.5;

  switch( i_probes )
  {
  case 1:
    //return boxMean( ii, x, y, major_len/0.886f );
    return felineImpl<2>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 2:
    return felineImpl<2>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 3:
    return felineImpl<3>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 4:
    return felineImpl<4>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 5:
    return felineImpl<5>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 6:
    return felineImpl<6>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 7:
    return felineImpl<7>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 8:
    return felineImpl<8>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  case 9:
    return felineImpl<9>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  default:
  {
	// limit number of probes to 10:
    minor_len = major_len * 4.0 / 21.0;
    return felineImpl<10>( ii, x, y, major_len, minor_len, major_x, major_y, major_len-minor_len );
  }
  }
}


}
}
#endif /* FELINE_H_ */
