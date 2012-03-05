/*
* Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "math_stuff.h"
#include "kernel2d.h"

#ifndef rgbd_features_filter_h_
#define rgbd_features_filter_h_

namespace cv
{

inline float gradX( const Mat1d &ii,
    const Mat1d &ii_depth_map, const cv::Mat_<uint64_t>& ii_depth_count,
    const cv::Matx33f& camera_matrix, int x, int y, float sp, float sw )
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
    const cv::Matx33f& camera_matrix, int x, int y, float sp, float sw )
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

/* Compute approximate affine Laplacian using a difference of rectangular integrals
*/
inline float dobAffine( const Mat1d &ii,
    const Mat1d &ii_depth_map, const cv::Mat_<uint64_t>& ii_depth_count,
    const cv::Matx33f& camera_matrix, int x, int y, float sp, float sw )
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

    Point2f major_axis( -grad[1], grad[0] );

    if ( major_axis.x != 0 && major_axis.y != 0 )
    {
      major_axis = major_axis * fastInverseLen(major_axis) * float(sp);

      if ( major_axis.y < 0 )
      {
        major_axis *= -1;
      }
    }

    const float SQRT_PI_2 = 0.886;

    // intersection of ellipsis with x/y axis
    float sw2 = sw*sw;
    float norm1 = sp * sw * SQRT_PI_2;
    float intersect_x = norm1 * fastInverseSqrt( sw2 + grad[0]*grad[0] );
    float intersect_y = norm1 * fastInverseSqrt( sw2 + grad[1]*grad[1] );

    // sizes of the four integral rectangles
    // sx1,sy1: top-left and bottom-right quadrant
    // sx2,sy2: top-right and bottom-left quadrant
    int sx1,sy1,sx2,sy2;

    if ( major_axis.x > 0 )
    {
      // major axis is in the top-right or bottom-left quadrant
      sx1 = std::max( intersect_x, major_axis.x );
      sy1 = std::max( intersect_y, major_axis.y );
      sx2 = intersect_x;
      sy2 = intersect_y;
    }
    else
    {
      // major axis is in the top-left or bottom-right quadrant
      sx1 = intersect_x;
      sy1 = intersect_y;
      sx2 = std::max( intersect_x, -major_axis.x );
      sy2 = std::max( intersect_y, major_axis.y );
    }

    if ( sx1 < 2 ) sx1=2;
    if ( sy1 < 2 ) sy1=2;
    if ( sx2 < 2 ) sx2=2;
    if ( sy2 < 2 ) sy2=2;

    float i1 = integrate ( ii, x-sx1  , x, y-sy1  , y );
    float i2 = integrate ( ii, x, x+sx2  , y-sy2  , y );
    float i3 = integrate ( ii, x-sx2  , x, y,   y+sy2 );
    float i4 = integrate ( ii, x, x+sx1  , y,   y+sy1 );

    float o1 = integrate ( ii, x-sx1*2, x, y-sy1*2, y );
    float o2 = integrate ( ii, x, x+sx2*2, y-sy2*2, y );
    float o3 = integrate ( ii, x-sx2*2, x, y, y+sy2*2 );
    float o4 = integrate ( ii, x, x+sx1*2, y, y+sy1*2 );

    float val = 4 * (i1+i2+i3+i4) - o1-o2-o3-o4;

    float area1 = sx1*sy1*8;
    float area2 = sx2*sy2*8;

    float val_norm = val / (area1+area2);
    return std::abs(val_norm);
  }

  return std::numeric_limits<float>::quiet_NaN();
}

/* Compute simple Laplacian (Difference Of Boxes)
 * Kernel size: 4s x 4s
 * Value range: 0..1
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0 -1 -1 -1 -1  0
 *  0    0 -1  3  3 -1  0
 *  1    0 -1  3  3 -1  0
 *  2    0 -1 -1 -1 -1  0
 *  3    0  0  0  0  0  0
*/
inline float dob( const Mat1d &ii, int x, int y, int s )
{
  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 2*s ) )
  {
    float val = 4 * integrate ( ii, x - s,  x + s, y - s, y + s )
                   - integrate ( ii, x - 2*s, x + 2*s, y - 2*s, y + 2*s );

    float val_norm = val / float(12.0*s*s);
    return std::abs(val_norm);
    /*
    if ( val_norm < 0.0 )
      return std::abs(val_norm);
    return 0;
    */
  }

  return std::numeric_limits<float>::quiet_NaN();
}

inline float laplaceAffine( const Mat1d &ii, int x, int y, float major, float minor, float angle )
{
  // 4.5 cells together have the size of major (=scale)
  int a = std::max(int(major/2.25f), 1);
  // check for boundary effects
  if ( !checkBounds( ii, x, y, 6*a ) ) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  // read mean intensities for 9x9 grid
  float values[9][9];
  integrateGridCentered<double,9>(ii, x - a/2, y - a/2, a, (float*)values);
  // convolve with ansisotrope laplace filter
  float response = sLaplaceKernelCache.convolve(values, minor/major, angle);
  // return normalized absolute response
  return std::abs(response) / float(a*a);
}

inline float laplace( const Mat1d &ii, int x, int y, int s )
{
  // 4.5 cells together have the size of major (=scale)
  int a = std::max(int(s/2.25f), 1);
  // check for boundary effects
  if ( !checkBounds( ii, x, y, 6*a ) ) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  // read mean intensities for 9x9 grid
  float values[9][9];
  integrateGridCentered<double,9>(ii, x - a/2, y - a/2, a, (float*)values);
  // convolve with isotrope laplace filter
  float response = sLaplaceKernel.convolve(values);
  // return normalized absolute response
  return std::abs(response) / float(a*a);
}

inline float princCurvRatio( const Mat1d &ii, int x, int y, int s )
{
  unsigned int a = std::max(int(s*0.5f), 1);

  if ( checkBounds( ii, x, y, 5*a ) )
  {
    unsigned int a = std::max(s / 2, 1);

    float values[9][9];
    for(int i=0; i<9; i++) {
        for(int j=0; j<9; j++) {
            values[i][j] = integrate(ii, x + a*(j-4), x + a*(j-3), y + a*(i-4), y + a*(i-3));
        }
    }

    float n = 1.0 / float(a*a);

    if ( x==300 && y==300 )
    {
      cv::Mat1f m1( 9,9, (float*)values[0] );
      std::ostringstream s;
      s << "s= " << s;
      showBig( 128, m1*n, s.str() );
    }

    float dxx = sDxxKernel.convolve(values) * n;
    float dyy = sDyyKernel.convolve(values) * n;
    float dxy = sDxyKernel.convolve(values) * n;

    float trace = dxx+dyy;
    float det = dxx*dyy - (dxy*dxy);

    if ( det <= 0 )
    {
      return std::numeric_limits<float>::max();
    }

    return trace*trace/det;
  }

  return std::numeric_limits<float>::quiet_NaN();
}

inline float princCurvRatioAffine( const Mat1d &ii, int x, int y, float major, float minor, float angle )
{
  unsigned int a = std::max(int(major*0.5f), 1);

  if ( checkBounds( ii, x, y, 5*a ) )
  {

    float values[9][9];
    for(int i=0; i<9; i++) {
        for(int j=0; j<9; j++) {
            values[i][j] = integrate(ii, x + a*(j-4), x + a*(j-3), y + a*(i-4), y + a*(i-3));
        }
    }

    float n = 1.0 / float(a*a);

    float dxx = sDxxKernelCache.convolve(values,minor/major, angle) * n;
    float dyy = sDyyKernelCache.convolve(values,minor/major, angle) * n;
    float dxy = sDxyKernelCache.convolve(values,minor/major, angle) * n;

    float trace = dxx+dyy;
    float det = dxx*dyy - (dxy*dxy);

    if ( det <= 0 )
    {
      return std::numeric_limits<float>::max();
    }

    return trace*trace/det;
    }

  return std::numeric_limits<float>::quiet_NaN();
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
inline float harris( const Mat1d &ii, int x, int y, int s )
{
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

#endif //rgbd_features_math_stuff_h_
