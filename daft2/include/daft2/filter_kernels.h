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

inline float gradX( const Mat1d &ii,
    const Mat1d &ii_depth_map, const cv::Mat_<uint64_t>& ii_depth_count,
    int x, int y, float sp, float sw, float min_sp )
{
  // sp : pixel scale
  // sw : world scale

  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 3*sp ) )
  {
    // depth gradient between (x+sp) and (x-sp)
    Vec2f grad;

    if ( !computeGradient( ii_depth_map, ii_depth_count, x, y, sp, grad ) )
      return std::numeric_limits<float>::quiet_NaN();

    return grad[0] / sw;
  }

  return std::numeric_limits<float>::quiet_NaN();
}

inline float gradY( const Mat1d &ii,
    const Mat1d &ii_depth_map, const cv::Mat_<uint64_t>& ii_depth_count,
    int x, int y, float sp, float sw, float min_sp )
{
  // sp : pixel scale
  // sw : world scale

  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 3*sp ) )
  {
    // depth gradient between (x+sp) and (x-sp)
    Vec2f grad;

    if ( !computeGradient( ii_depth_map, ii_depth_count, x, y, sp, grad ) )
      return std::numeric_limits<float>::quiet_NaN();

    return grad[1] / sw;
  }

  return std::numeric_limits<float>::quiet_NaN();
}

/* Compute approximate affine Gaussian using rectangular integrals */
inline float boxAffine( const Mat1d &ii, Vec2f &grad,
    int x, int y, float sp, float sw, float min_sp )
{
  // sp : pixel scale
  // sw : world scale

  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 3*sp ) )
  {
    Point2f major_axis( -grad[1], grad[0] );

    if ( major_axis.x != 0 || major_axis.y != 0 )
    {
      major_axis = major_axis * fastInverseLen(major_axis) * float(sp);

      if ( major_axis.y < 0 )
      {
        major_axis *= -1;
      }
    }

    static const float SQRT_PI_2 = 0.886;

    // intersection of ellipsis with x/y axis
    const float sw2 = sw*sw;
    const float norm1 = sp * sw * SQRT_PI_2;
    float intersect_x = norm1 * fastInverseSqrt( sw2 + grad[0]*grad[0] );
    float intersect_y = norm1 * fastInverseSqrt( sw2 + grad[1]*grad[1] );

    intersect_x += 0.5;
    intersect_y += 0.5;
    major_axis.y += 0.5;

    // sizes of the four integral rectangles
    // sx1,sy1: top-left and bottom-right quadrant
    // sx2,sy2: top-right and bottom-left quadrant
    int sx1,sy1,sx2,sy2;

    if ( major_axis.x > 0 )
    {
      major_axis.x += 0.5;
      // major axis is in the top-right or bottom-left quadrant
      sx1 = std::max( intersect_x, major_axis.x );
      sy1 = std::max( intersect_y, major_axis.y );
      sx2 = intersect_x;
      sy2 = intersect_y;
    }
    else
    {
      major_axis.x -= 0.5;
      sx1 = intersect_x;
      // major axis is in the top-left or bottom-right quadrant
      sy1 = intersect_y;
      sx2 = std::max( intersect_x, -major_axis.x );
      sy2 = std::max( intersect_y, major_axis.y );
    }

    if ( sx1 < min_sp || sy1 < min_sp || sx2 < min_sp || sy2 < min_sp )
    {
      return std::numeric_limits<float>::quiet_NaN();
    }

    float i1 = integrate ( ii, x-sx1  , x, y-sy1  , y );
    float i2 = integrate ( ii, x, x+sx2  , y-sy2  , y );
    float i3 = integrate ( ii, x-sx2  , x, y,   y+sy2 );
    float i4 = integrate ( ii, x, x+sx1  , y,   y+sy1 );

    float val = i1+i2+i3+i4;

    float area1 = sx1*sy1*2.0f;
    float area2 = sx2*sy2*2.0f;

    float val_norm = val / (area1+area2);
    return val_norm * 0.73469f; // normalize to same value as the laplace filter
  }

  return std::numeric_limits<float>::quiet_NaN();
}


/* Compute box mean with sub-integer scale interpolation */
inline float box( const Mat1d &ii, int x, int y, float s )
{
  // normalizes to same max value as the gauss filter
  static const float NORM_1 = 0.73469f;
  // make the box which cover the same area as a circle withradius s
  s *= 0.886f;
  int si = int(s);
  int si1 = si+1;
  if (!checkBounds( ii, x, y, si+1 ) )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float v1 = integrate ( ii, x - si,  x + si, y - si, y + si ) / float(4*si*si);
  float v2 = integrate ( ii, x - si1,  x + si1, y - si1, y + si1 ) / float(4*si1*si1);
  return interpolateLinear(s - float(si), v1, v2) * NORM_1;
}

/** integer as parameter */
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
inline float gaussAffine( const Mat1d &ii, Vec2f& grad, int x, int y, float sp, float sw, float min_sp )
{
    float angle, major, minor;
    Point3f normal;
    bool ok = getAffine(grad, x, y, sp, sw, angle, major, minor, normal);
    // break if gradient can not be computed
    // or minor axis too small
    if(!ok || minor < min_sp) {
    	return std::numeric_limits<float>::quiet_NaN();
    }

  float a = 0.5893f * major; // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float ratio = minor / major;
  float v1 = gaussAffineImpl(ii, x, y, ai, ratio, angle);
  float v2 = gaussAffineImpl(ii, x, y, ai + 1, ratio, angle);
  return interpolateLinear(t, v1, v2);
}

/** integer as parameter */
inline float gaussImpl( const Mat1d &ii, int x, int y, int a )
{
  // check for boundary effects
  if ( !checkBounds( ii, x, y, 6*a ) ) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  // read mean intensities for 9x9 grid
  float values[9][9];
  integrateGridCentered<double,9>(ii, x, y, a, (float*)values);
  // convolve with isotrope laplace filter
  float response = sGaussKernel.convolve(values);
  // return normalized absolute response
  return std::abs(response) / float(a*a);
}

/** float as parameter and interpolates */
inline float gauss( const Mat1d &ii, int x, int y, float s )
{
  float a = 0.5893f * s; // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float v1 = gaussImpl(ii, x, y, ai);
  float v2 = gaussImpl(ii, x, y, ai + 1);
  return interpolateLinear(t, v1, v2);
}

/** integer as parameter */
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

/** float as parameter and interpolates */
inline float princCurvRatio( const Mat1d &ii, int x, int y, float s )
{
  float a = 0.5893f * s; // sqrt(2)/1.2/2
  int ai = int(a);
  float t = a - float(ai);
  float v1 = princCurvRatioImpl(ii, x, y, ai);
  float v2 = princCurvRatioImpl(ii, x, y, ai + 1);
  return interpolateLinear(t, v1, v2);
}

/** integer as parameter */
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

/** float as parameter and interpolates */
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
