/*
* Copyright (C) 2011 David Gossow
*/

#ifndef rgbd_features_math_stuff_h_
#define rgbd_features_math_stuff_h_

#include <opencv2/core/core.hpp>

namespace cv
{

// Compute the integral of the rectangle (start_x,start_y),(end_x,end_y)
// using the given integral image
inline float integrate( const Mat1d &ii, int start_x, int end_x, int start_y, int end_y )
{
  assert( start_x>=0 );
  assert( end_x>start_x );
  assert( end_x<ii.cols );
  assert( start_y>=0 );
  assert( end_y>start_y );
  assert( end_y<ii.rows );
  return ii(end_y,end_x) + ii(start_y,start_x) - ii(end_y,start_x) - ii(start_y,end_x);
}

// compute the area of the given rect
inline double area(int start_x, int end_x, int start_y, int end_y)
{
  return (end_y-start_y) * (end_y-start_y);
}

// return false if the square at (x,y) with size s*2 intersect the image border
inline bool checkBounds ( Mat1d ii, int x, int y, int s )
{
  return ( (x > s) && (x + s < ii.cols) && (y > s) && (y + s < ii.rows) );
}

// compute 3d point from pixel position, depth and camera intrinsics
// @param f_inv: 1/f
// @param cx,cy optical center
// @param u,v pixel coords
// @param p output point in 3d
inline void getPt3d( float f_inv, float cx, float cy, float u, float v, float z, Point3f& p )
{
  float zf = z*f_inv;
  p.x = zf * (u-cx);
  p.y = zf * (v-cy);
  p.z = z;
}

inline void getPt2d( const Point3f& p, float f, float cx, float cy, Point2f& v )
{
  v.x = p.x * f / p.z + cx;
  v.y = p.y * f / p.z + cy;
}

// approximate 1 / sqrt(x) with accuracy ~2%
inline float fastInverseSqrt(float x)
{
  uint32_t i = *((uint32_t *)&x);            // evil floating point bit level hacking
  i = 0x5f3759df - (i >> 1);                // use a magic number
  float s = *((float *)&i);                // get back guess
  return s * (1.5f - 0.5f * x * s * s);    // one newton iteration
}

// compute 1/length(p)
inline float fastInverseLen( const Point2f& p )
{
  return fastInverseSqrt( p.x*p.x + p.y*p.y );
}

// compute 1/length(p)
inline float fastInverseLen( const Point3f& p )
{
  return fastInverseSqrt( p.x*p.x + p.y*p.y + p.z*p.z );
}

// compute depth gradient
inline bool computeGradient( const Mat1f &depth_map,
    int x, int y, float sp, float sw, Vec2f& grad )
{
  // get depth values from image
  float d_center = depth_map(y,x);
  float d_xp = depth_map(y,x+sp*2);
  float d_yp = depth_map(y+sp*2,x);
  float d_xn = depth_map(y,x-sp*2);
  float d_yn = depth_map(y-sp*2,x);

  if ( isnan(d_center) || isnan(d_xp) || isnan(d_yp) || isnan(d_xn) || isnan(d_yn) )
  {
    return false;
  }

  float dxx = d_xp - 2*d_center + d_xn;
  float dyy = d_yp - 2*d_center + d_yn;

  // test for local planarity
  if ( dxx*dxx + dyy*dyy > sw*sw*10 )
  {
    return false;
  }

// depth gradient between (x+sp) and (x-sp)
  grad[0] = (d_xp - d_xn)*0.25;
  grad[1] = (d_yp - d_yn)*0.25;
  return true;
}


// sp : pixel scale
// sw : world scale
inline bool getAffine(
    const Mat1d &ii,
    const Mat1f &depth_map,
    int x, int y,
    float sp, float sw,
    float &angle, float &major, float &minor,
    Point3f& normal )
{
  // the depth gradient
  Vec2f grad;

  if ( !checkBounds( ii, x, y, sp*2 )  ||
       !computeGradient( depth_map, x, y, sp, sw, grad ) )
  {
    major = minor = sp;
    angle = 0;
    return false;
  }

  // if the gradient is 0, make circle
  if ( grad[0] == 0 && grad[1] == 0 )
  {
    major = minor = sp;
    angle = 0;
    return true;
  }

  // gradient, normalized to length=1

  // len(grad)^2
  float grad_len_2 = grad[0]*grad[0]+grad[1]*grad[1];
  Vec2f grad_norm = grad * fastInverseSqrt( grad_len_2 );

  // compute the minor axis length
  minor = 4 * sp * fastInverseSqrt( grad_len_2 / (sw*sw) + 1.0f );
  major = 4 * sp;
  angle = atan2( grad_norm[0], -grad_norm[1] );

  normal.x = grad[0] / sw;
  normal.y = grad[1] / sw;
  normal.z = -1.0f;
  normal = normal * (1.0 / cv::norm( normal ));

  return true;
}

}

#endif
