/*
 * gauss9x9.h
 *
 *  Created on: May 24, 2012
 *      Author: gossow
 */

#ifndef GAUSS9X9_H_
#define GAUSS9X9_H_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "stuff.h"
#include "kernel2d.h"


namespace cv
{
namespace daft2
{

inline float gaussAffineImpl( const Mat1d &ii, int x, int y, int a, float ratio, float angle )
{
  // check for boundary effects
  if ( !checkBounds( ii, x, y, 7*a ) ) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  // read mean intensities for 9x9 grid
  float values[9][9];
  integrateGridCentered<double,9>(ii, x, y, a, (float*)values);
  // convolve with ansisotropic laplace filter
  float response = sGaussianKernelCache.convolve(values, ratio, angle);
  // return normalized absolute response
  return std::abs(response) / float(a*a);
}



/** float as parameter and interpolates */
inline float gaussAffine( const Mat1d &ii,
    int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y )
{
  float sp=major_len;
  float minor_ratio = minor_len / major_len;

  float angle = std::atan2( major_y, major_x );

  float a = 0.5 * 0.5893f * sp * (1.2/2.4); // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float v1 = gaussAffineImpl(ii, x, y, ai, minor_ratio, angle);
  float v2 = gaussAffineImpl(ii, x, y, ai + 1, minor_ratio, angle);
  return interpolateLinear(t, v1, v2) / 0.7 * (1.2/2.4);
}

}
}

#endif /* GAUSS9X9_H_ */
