/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT_STUFF_H__
#define __DAFT_STUFF_H__

#include <opencv2/core/core.hpp>

#include <cmath>

namespace cv
{
namespace daft
{

const float nan = std::numeric_limits<float>::quiet_NaN();

Matx33f inline pointCovariance(const std::vector<Vec3f>& points)
{
    float xx=0.0f, xy=0.0f, xz=0.0f, yy=0.0f, yz=0.0f, zz=0.0f;
    for( std::vector<Vec3f>::const_iterator it=points.begin(); it!=points.end(); ++it)
    {
        const Vec3f& p = *it;
        float x = p[0];
        float y = p[1];
        float z = p[2];
        xx += x*x;
        xy += x*y;
        xz += x*z;
        yy += y*y;
        yz += y*z;
        zz += z*z;
    };
    Matx33f A;
    A << xx, xy, xz, xy, yy, yz, xz, yz, zz;
    return A;
}

/** Fits a plane into points and returns the plane normal */
Vec3f inline fitNormal(const std::vector<Vec3f>& points)
{
    Matx33f A = pointCovariance(points);
    Mat1f eigen_vals;
    Mat1f eigen_vecs;
    cv::eigen(A, eigen_vals,eigen_vecs);
    Vec3f normal( eigen_vecs[2][0], eigen_vecs[2][1], eigen_vecs[2][2] );
    if ( normal[2] > 0.0 ) normal *= -1.0;
    return normal;
}



// create integral image & convert type T1 to T2, multiplying with factor
template<typename T1,typename T2>
void integral2( const cv::Mat_<T1>& m_in, cv::Mat_<T2>& m_out, T2 factor=1.0 )
{
  m_out.create(m_in.rows + 1, m_in.cols + 1);
  for (int y = 0; y < m_out.rows; y++)
  {
    double row_sum = 0;
    for (int x = 0; x < m_out.cols; x++)
    {
      if (x == 0 || y == 0) {
        m_out[y][x] = 0;
      } else {
        T2 val = factor * static_cast<T2>(m_in[y - 1][x - 1]);
        if ( isnan(val) ) val=0;
        row_sum += val;
        m_out[y][x] = row_sum + m_out[y - 1][x];
      }
    }
  }
}

inline void depthIntegral( const cv::Mat1f& depth_map, cv::Mat1d& ii_depth_map, cv::Mat_<uint32_t>& ii_depth_count )
{
  ii_depth_map.create(depth_map.rows + 1, depth_map.cols + 1);
  ii_depth_count.create(depth_map.rows + 1, depth_map.cols + 1);

  for (int y = 0; y < depth_map.rows + 1; y++) {
    double row_sum = 0;
    double row_count = 0;
    for (int x = 0; x < depth_map.cols + 1; x++) {
      if (x == 0 || y == 0) {
        ii_depth_map[y][x] = 0;
        ii_depth_count[y][x] = 0;
      } else {
        float depth = depth_map[y - 1][x - 1];
        if (!isnan(depth)) {
          row_sum += depth;
          row_count += 1;
        }
        ii_depth_map[y][x] = row_sum + ii_depth_map[y - 1][x];
        ii_depth_count[y][x] = row_count + ii_depth_count[y - 1][x];
      }
    }
  }
}

// Compute the integral of the rectangle (start_x,start_y),(end_x,end_y)
// using the given integral image
template<typename T>
inline T integrate( const Mat_<T> &ii, int start_x, int start_y, int end_x, int end_y )
{
  assert( start_x>=0 );
  assert( end_x>start_x );
  assert( end_x<ii.cols );
  assert( start_y>=0 );
  assert( end_y>start_y );
  assert( end_y<ii.rows );
  return ii(start_y,start_x) + ii(end_y,end_x) - ii(end_y,start_x) - ii(start_y,end_x);
}

// compute the area of the given rect
inline double area(int start_x, int end_x, int start_y, int end_y)
{
  return (end_y-start_y) * (end_y-start_y);
}

// return false if the square at (x,y) with size s*2 intersect the image border
template<typename T>
inline bool checkBounds ( const Mat_<T> &ii, int x, int y, int s )
{
  return ( (s > 0) && (x > s) && (x + s < ii.cols) && (y > s) && (y + s < ii.rows) );
}

inline float getScale( float fac, float base_scale )
{
  return fac*base_scale;
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


/** compute depth gradient
 * @param sp step width in projected pixel
 */
inline bool computeGradient(
    const Mat1f &depth_map, int x, int y, float sp, Vec2f& grad)
{
  int sp_int = int(sp+0.5f);

  if ( !checkBounds( depth_map, x, y, sp_int ) )
  {
    grad[0] = grad[1] = nan;
    return false;
  }

  // get depth values from image
  const float d_center = depth_map(y,x);
  const float d_xp = depth_map(y,x+sp_int);
  const float d_yp = depth_map(y+sp_int,x);
  const float d_xn = depth_map(y,x-sp_int);
  const float d_yn = depth_map(y-sp_int,x);

  if ( isnan(d_center) || isnan(d_xp) || isnan(d_yp) || isnan(d_xn) || isnan(d_yn) )
  {
    grad[0] = grad[1] = nan;
    return false;
  }

  const float fac = 0.5*sp/float(sp_int);
  grad[0] = (d_xp - d_xn)*fac;
  grad[1] = (d_yp - d_yn)*fac;
  return true;
}

}
}

#endif
