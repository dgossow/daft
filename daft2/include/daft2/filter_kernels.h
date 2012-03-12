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

template<int Accuracy>
inline float gaussAffineX( const Mat1d &ii, Vec2f &grad,
    int x, int y, float sp, float sw, float min_sp )
{
  // The roots of the 1-d difference of gaussians
  // e^(-x^2/8)/(2 sqrt(2 pi)) - e^(-x^2/2)/sqrt(2 pi)
  // are at +- 1.3595559868917
  static const float ROOT_DOG_INV = 0.735534255;

  // intersection of ellipsis with x/y axis
  const float sw2 = sw*sw;
  const float norm1 = sp * sw;
  // x intersection of affine ellipse with radius sp
  sp = norm1 * fastInverseSqrt( sw2 + grad[0]*grad[0] );
  //std::cout << " sp " << sp << " inter_x " << intersect_x << std::endl;

  // integer pixel scale
  const int spi = std::max( int(sp / float(Accuracy)), 1 );
  const int num_steps = int(std::ceil( sp/float(spi)*2.0f ));
  const int win_size = num_steps * spi;
  //const float alpha = float(win_size) / sp*2;

  if ( checkBounds( ii, x, y, win_size ) )
  {
    // this mu makes the zero crossing of the difference-of-gaussians be at +/- sp
    // g = e^(-(x-μ)^2/(2 σ^2))/(sqrt(2 π) σ)
    const float sigma = ROOT_DOG_INV * sp / 2; //* intersect_x;

    static int last_ws = 0;

#if 0
    if ( last_ws != num_steps )
      std::cout << "-- sp " << sp << " sigma " << sigma << std::endl;
#endif

    float val = 0;
    float sum_gauss = 0;

    //static const float expCache[MAX_NSTEPS] = {};

    for ( int t = -win_size; t<win_size; t+=spi )
    {
      const float t2 = ((float)t+0.5f*spi);
      // g(x) = e^( -(x)^2/(2 σ^2) ) / (sqrt(2 π) σ)
      static const float n = sqrt( 2.0 * M_PI ) * sigma;
      const float g = std::exp( -t2*t2 / (2.0f*sigma*sigma) ) / n;
#if 0
      if ( last_ws != num_steps )
        std::cout << g << " ";
#endif
      const int x2 = x+t;
      val += g * float(integrate( ii, x2, x2+spi, y, y+1 )) / float(spi);
      sum_gauss += g;
    }
#if 0
    if ( last_ws != num_steps )
      std::cout << std::endl;
    //std::cout << std::endl << " sum/mu = " << sv/mu << " mu =  " << mu << std::endl;
    if ( last_ws != num_steps )
      std::cout<< " val " << val << std::endl << " num_steps = " << num_steps << " sum_gauss = " << sum_gauss << std::endl;
    last_ws = num_steps;
#endif

    //return val / float(spi*win_size*2);
    return val / sum_gauss;
  }

  return std::numeric_limits<float>::quiet_NaN();
}

template<int Accuracy>
inline float gaussAffineY( const Mat1f &ii_y, const Mat1f &ii_y_count, Vec2f &grad,
    int x, int y, float sp, float sw, float min_sp )
{
  // The roots of the 1-d difference of gaussians
  // e^(-x^2/8)/(2 sqrt(2 pi)) - e^(-x^2/2)/sqrt(2 pi)
  // are at +- 1.3595559868917
  static const float ROOT_DOG_INV = 0.735534255;

  // compute major axis
  Vec2f major_axis( grad * fastInverseSqrt( grad[0]*grad[0]+grad[1]*grad[1] ) * sp );
  // scale in y-direction
  sp = major_axis[1];

  // integer pixel scale
  const int spi = std::max( int(sp / float(Accuracy)), 1 );
  const int num_steps = std::max( int(std::ceil( sp/float(spi)*2.0f )), 1 );
  const int win_size = num_steps * spi;
  //const float alpha = float(win_size) / sp*2;

  if ( checkBounds( ii_y, x, y, win_size ) )
  {
    // this mu makes the zero crossing of the difference-of-gaussians be at +/- sp
    // g = e^(-(x-μ)^2/(2 σ^2))/(sqrt(2 π) σ)
    const float sigma = ROOT_DOG_INV * sp / 2; //* intersect_x;

    static int last_ws = 0;

#if 0
    if ( last_ws != num_steps )
      std::cout << "-- sp " << sp << " sigma " << sigma << std::endl;
#endif

    float val = 0;
    float sum_gauss = 0;

    //static const float expCache[MAX_NSTEPS] = {};

    for ( int t = -win_size; t<win_size; t+=spi )
    {
      const float t2 = ((float)t+0.5f*spi);
      // g(x) = e^( -(x)^2/(2 σ^2) ) / (sqrt(2 π) σ)
      static const float n = sqrt( 2.0 * M_PI ) * sigma;
      const float g = std::exp( -t2*t2 / (2.0f*sigma*sigma) ) / n;
#if 0
      if ( last_ws != num_steps )
        std::cout << g << " ";
#endif
      const int y2 = y+t;
      const int x_offs = t2 / float(win_size);
      const int x2 = x+x_offs;
      if ( last_ws != num_steps )
      std::cout << "x2 " << x2 << " y2 " << y2 << std::endl;
      val += g * ( ii_y[y2+spi][x] - ii_y[y2][x] ) / float( ii_y_count[y2+spi][x] - ii_y_count[y2][x] );
      sum_gauss += g;
    }
#if 1
    if ( last_ws != num_steps )
      std::cout << std::endl;
    //std::cout << std::endl << " sum/mu = " << sv/mu << " mu =  " << mu << std::endl;
    if ( last_ws != num_steps )
      std::cout<< " val " << val << std::endl << " num_steps = " << num_steps << " sum_gauss = " << sum_gauss << std::endl;
    last_ws = num_steps;
#endif

    //return val / float(spi*win_size*2);
    return val / sum_gauss;
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
  if ( !checkBounds( ii, x, y, 6*a ) ) {
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

/*
 *  0  0  0  0
 *  0  1  1  0
 *  0  1  1  0
 *  0  0  0  0
 */
inline float iiMean( const Mat1d &ii, int x, int y, int s )
{
  return integrate ( ii, x - s,  x + s, y - s, y + s ) / float(4*s*s);
}

/*
 *  0  0  0  0
 * -1 -1  1  1
 * -1 -1  1  1
 *  0  0  0  0
 */
inline float iiDx( const Mat1d &ii, int x, int y, int s )
{
    return ( integrate ( ii, x,  x + 2*s, y - s, y + s )
           - integrate ( ii, x - 2*s,  x, y - s, y + s ) ) / float(4*s*s);
  return 0;
}
inline float iiDy( const Mat1d &ii, int x, int y, int s )
{
    return ( integrate ( ii, x - s,  x + s, y, y + 2*s )
           - integrate ( ii, x - s,  x + s, y - 2*s, y ) ) / float(4*s*s);
  return 0;
}



/* Compute Harris corner measure h(x,y)
 * Value range: 0..1
*/
inline float harris( const Mat1d &ii, int x, int y, float s_real )
{
  int s = s_real;
  // FIXME interpolate!!!
  if ( checkBounds( ii, x, y, 4*s ) )
  {
    double sum_dxdx=0;
    double sum_dydy=0;
    double sum_dxdy=0;

    // dx and dy have range -4s² .. 4s²
    double norm = 0.25 / double(s*s);
    int s2 = s*2;

    for ( int x2 = x-s; x2 <= x+s; x2 += s )
    {
      for ( int y2 = y-s; y2 <= y+s; y2 += s )
      {
        double dx = ( - integrate ( ii, x2-s2, x2,    y2-s,  y2+s  )
                      + integrate ( ii, x2,    x2+s2, y2-s,  y2+s  ) ) * norm;
        double dy = ( - integrate ( ii, x2-s,  x2+s,  y2-s2, y2    )
                      + integrate ( ii, x2-s,  x2+s,  y2,    y2+s2 ) ) * norm;
        sum_dxdx += dx * dx;
        sum_dydy += dy * dy;
        sum_dxdy += dx * dy;
      }
    }

    double trace = ( sum_dxdx + sum_dydy );
    double det = (sum_dxdx * sum_dydy) - (sum_dxdy * sum_dxdy);

    return det - 0.1 * (trace * trace);
  }

  return std::numeric_limits<double>::quiet_NaN();
}


}
}

#endif //rgbd_features_math_stuff_h_
