/*
* Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>

#include <iostream>

#include "math_stuff.h"

#ifndef rgbd_features_filter_h_
#define rgbd_features_filter_h_

namespace cv
{



/* Compute 2nd discrete derivative f_xx(x,y)
 * Result is not normalized!
 * Kernel size: 4s x 4s.
 * Value range: -8s²..8s²
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  0  0  0  0  0
 *  0    0  1 -1 -1  1  0
 *  1    0  1 -1 -1  1  0
 *  2    0  0  0  0  0  0
 *  3    0  0  0  0  0  0
*/
inline double dxx( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 2*s, x + 2*s, y - s, y + s )
  -2.0 * integral ( ii, x - s,   x + s,   y - s, y + s );
}

/* Compute 2nd discrete derivative f_yy(x,y)
 * Analogous to dxx(ii,x,y).
 */
inline double dyy( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - s, x + s, y - 2*s, y + 2*s )
  -2.0 * integral ( ii, x - s, x + s,   y - s, y + s   );
}

/* Compute 2nd discrete derivative f_xy(x,y)
 * Kernel size: 4s x 4s
 * Value range: -8s²..8s²
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  1  1 -1 -1  0
 *  0    0  1  1 -1 -1  0
 *  1    0 -1 -1  1  1  0
 *  2    0 -1 -1  1  1  0
 *  3    0  0  0  0  0  0
*/
inline double dxy( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 2*s, x, y - 2*s, y )
      - integral ( ii, x, x + 2*s, y - 2*s, y )
      - integral ( ii, x - 2*s, x, y, y + 2*s )
      + integral ( ii, x, x + 2*s, y, y + 2*s );
}

/* Compute approximate affine Laplacian using a difference of rectangular integrals
*/
inline double dobAffine( const Mat1d &ii, const Mat1f &depth_map, const cv::Matx33f& camera_matrix, int x, int y, float sp, float sw )
{
  // sp : pixel scale
  // sw : world scale

  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 3*sp ) )
  {
    // depth gradient between (x+sp) and (x-sp)
    Vec2f grad;

    if ( !computeGradient( depth_map, x, y, sp, sw, grad ) )
      return std::numeric_limits<double>::quiet_NaN();
/*
    // get depth values from image
    float d_xp = depth_map(y,x+sp);
    float d_yp = depth_map(y+sp,x);
    float d_xn = depth_map(y,x-sp);
    float d_yn = depth_map(y-sp,x);

    if ( isnan(d_xp) || isnan(d_yp) || isnan(d_xn) || isnan(d_yn) )
      return std::numeric_limits<double>::quiet_NaN();

    // depth gradient between (x+sp) and (x-sp)
    grad[0] = (d_xp - d_xn)*0.5;
    grad[1] = (d_yp - d_yn)*0.5;
*/
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

    // h is in the top-right or bottom-left quadrant
    if ( major_axis.x > 0 )
    {
      sx1 = std::max( intersect_x, major_axis.x );
      sy1 = std::max( intersect_y, major_axis.y );
      sx2 = intersect_x;
      sy2 = intersect_y;
    }
    else
    {
      sx1 = intersect_x;
      sy1 = intersect_y;
      sx2 = std::max( intersect_x, -major_axis.x );
      sy2 = std::max( intersect_y, major_axis.y );
    }

    if ( sx1 < 1 ) sx1=1;
    if ( sy1 < 1 ) sy1=1;
    if ( sx2 < 1 ) sx2=1;
    if ( sy2 < 1 ) sy2=1;

    float i1 = integral ( ii, x-sx1  , x, y-sy1  , y );
    float i2 = integral ( ii, x, x+sx2  , y-sy2  , y );
    float i3 = integral ( ii, x-sx2  , x, y,   y+sy2 );
    float i4 = integral ( ii, x, x+sx1  , y,   y+sy1 );

    float o1 = integral ( ii, x-sx1*2, x, y-sy1*2, y );
    float o2 = integral ( ii, x, x+sx2*2, y-sy2*2, y );
    float o3 = integral ( ii, x-sx2*2, x, y, y+sy2*2 );
    float o4 = integral ( ii, x, x+sx1*2, y, y+sy1*2 );

    float val = 4 * (i1+i2+i3+i4) - o1-o2-o3-o4;

    float area1 = sx1*sy1*8;
    float area2 = sx2*sy2*8;

    float val_norm = val / (area1+area2);
    return std::abs(val_norm);
  }

  return std::numeric_limits<double>::quiet_NaN();
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
inline double dob( const Mat1d &ii, int x, int y, int s )
{
  //std::cout << x << " " << y << "   " << s << " * 2 = " << 2*s << std::endl;
  if ( checkBounds( ii, x, y, 2*s ) )
  {
    double val = 4 * integral ( ii, x - s,  x + s, y - s, y + s )
                   - integral ( ii, x - 2*s, x + 2*s, y - 2*s, y + 2*s );

    double val_norm = val / double(12.0*s*s);
    return std::abs(val_norm);
    /*
    if ( val_norm < 0.0 )
      return std::abs(val_norm);
    return 0;
    */
  }

  return std::numeric_limits<double>::quiet_NaN();
}



/* Compute Laplacian l(x,y) = f_xx(x,y)+f_yy(x,y)
 * Kernel size: 4s x 4s
 * Value range: 0..1
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  0  1  1  0  0
 *  0    0  1 -2 -2  1  0
 *  1    0  1 -2 -2  1  0
 *  2    0  0  1  1  0  0
 *  3    0  0  0  0  0  0
*/
inline double laplace( const Mat1d &ii, int x, int y, int s )
{
  if ( checkBounds( ii, x, y, 2*s ) )
  {
    double v = integral ( ii, x - 2*s, x + 2*s, y - s,   y + s   )
            +  integral ( ii, x - s,     x + s, y - 2*s, y + 2*s )
        -4.0 * integral ( ii, x - s,     x + s, y - s,   y + s   );
    return std::abs( v ) / (32.0*double(s*s));
  }
  return std::numeric_limits<double>::quiet_NaN();
}



/* Compute Harris corner measure h(x,y)
 * Value range: 0..1
*/
inline double harris( const Mat1d &ii, int x, int y, int s )
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
        double dx = ( - integral ( ii, x2-s2, x2,    y2-s,  y2+s  )
                      + integral ( ii, x2,    x2+s2, y2-s,  y2+s  ) ) * norm;
        double dy = ( - integral ( ii, x2-s,  x2+s,  y2-s2, y2    )
                      + integral ( ii, x2-s,  x2+s,  y2,    y2+s2 ) ) * norm;
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
