/*
* Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>

#include <iostream>


#ifndef rgbd_features_filter_h_
#define rgbd_features_filter_h_

namespace cv
{

// Compute the integral of the rectangle (start_x,start_y),(end_x,end_y)
// using the given integral image
inline double integral( const Mat1d &ii, int start_x, int end_x, int start_y, int end_y )
{
  assert( start_x>0 );
  assert( start_y>0 );
  assert( start_x<ii.cols );
  assert( start_y<ii.rows );
  assert( end_x>0 );
  assert( end_y>0 );
  assert( end_x<ii.cols );
  assert( end_y<ii.rows );
  return ii(end_y,end_x) + ii(start_y,start_x) - ii(end_y,start_x) - ii(start_y,end_x);
}

inline double area(int start_x, int end_x, int start_y, int end_y)
{
  return (end_y-start_y) * (end_y-start_y);
}

// return false if the square at (x,y) with size s*2 intersect the image border
inline bool checkBounds ( Mat1d ii, int x, int y, int s )
{
  return ( (x > s) && (x + s < ii.cols) && (y > s) && (y + s < ii.rows) );
}


/* Compute 2nd discrete derivative f_xx(x,y)
 * Result is not normalized!
 * Kernel size: 6s x 4s.
 * Value range: -16s²..16s²
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    1  1 -2 -2  1  1
 *  0    1  1 -2 -2  1  1
 *  1    1  1 -2 -2  1  1
 *  2    1  1 -2 -2  1  1
 *  3    0  0  0  0  0  0
*/
inline double dxx( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 3*s,   x + 3*s, y - 2*s, y + 2*s )
      - 3.0 * integral ( ii, x - s,  x + s, y - 2*s, y + 2*s );
}

/* Compute 2nd discrete derivative f_yy(x,y)
 * Analogous to dxx(ii,x,y).
 */
inline double dyy( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 2*s, x + 2*s, y - 3*s, y + 3*s )
      - 3.0 * integral ( ii, x - 2*s,    x + 2*s, y - s, y + s );
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
 * -2    0  1  1  1  1  0
 * -1    1  2 -1 -1  2  1
 *  0    1 -1 -4 -4 -1  1
 *  1    1 -1 -4 -4 -1  1
 *  2    1  2 -1 -1  2  1
 *  3    0  1  1  1  1  0
*/
inline double laplace( const Mat1d &ii, int x, int y, int s )
{

  if ( checkBounds( ii, x, y, 2*s ) )
  {
    double ii_dxx = dxx( ii, x, y, s );
    double ii_dyy = dyy( ii, x, y, s );

    return std::abs( (ii_dxx + ii_dyy) / (32.0*double(s*s)) );
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
