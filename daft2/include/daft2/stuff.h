/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_STUFF_H__
#define __DAFT2_STUFF_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cv
{

/** Interpolates linerarly between v1 and v2 given a percentage t */
template<typename T>
inline T interpolateLinear(T t, T v1, T v2)
{
  return (static_cast<T>(1) - t) * v1 + t * v2;
}

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
        m_out[y][x] = row_sum + m_out[y - 1][x];
        row_sum += val;
      }
    }
  }
}

// Compute the integral of the rectangle (start_x,start_y),(end_x,end_y)
// using the given integral image
template<typename T>
inline T integrate( const Mat_<T> &ii, int start_x, int end_x, int start_y, int end_y )
{
  assert( start_x>=0 );
  assert( end_x>start_x );
  assert( end_x<ii.cols );
  assert( start_y>=0 );
  assert( end_y>start_y );
  assert( end_y<ii.rows );
  return ii(start_y,start_x) + ii(end_y,end_x) - ii(end_y,start_x) - ii(start_y,end_x);
}

/** Gets integration value of NxN cells of size step x step in a grid starting with point (start_x,start_y)*/
template<typename T, int N>
inline void integrateGridCentered( const Mat_<T> &ii, int start_x, int start_y, int step, float* values) {
//  for(int i=0; i<N; i++) {
//    for(int j=0; j<N; j++) {
//      int x = start_x + step*(j-M);
//      int y = start_y + step*(i-M);
//      values[i*N + j] = integrate(ii, x, x + step, y, y + step);
//    }
//  }
  // look up values in integral image
  float lookup[N+1][N+1];
  for(int i=0; i<N+1; i++) {
    for(int j=0; j<N+1; j++) {
      int x = start_x + step*j - (step*N)/2;
      int y = start_y + step*i - (step*N)/2;
      lookup[i][j] = ii(y, x);
    }
  }
  // compute cell integrals
  for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
      values[i*N + j] = lookup[i][j] + lookup[i+1][j+1] - lookup[i+1][j] - lookup[i][j+1];
    }
  }
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


inline float meanDepth(const Mat1d &ii_depth_map,
    const cv::Mat_<uint64_t>& ii_depth_count,
    int x, int y, int sp_int )
{
  float nump = float(integrate( ii_depth_count, x-sp_int, x+sp_int, y-sp_int, y+sp_int ));
  if ( nump == 0 )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return integrate( ii_depth_map, x-sp_int, x+sp_int, y-sp_int, y+sp_int ) / nump;
}

/** compute depth gradient
 * @param sp step width in projected pixel
 */
inline bool computeGradient(
    const Mat1d &ii_depth_map, const cv::Mat_<uint64_t>& ii_depth_count,
    int x, int y, float sp, Vec2f& grad)
{
  int sp_int = int(sp+0.5f);

  if ( sp_int < 6 )
  {
    sp_int = 6;
  }

  if ( !checkBounds( ii_depth_count, x, y, sp_int*2 ) )
  {
    grad[0] = std::numeric_limits<float>::quiet_NaN();
    grad[1] = std::numeric_limits<float>::quiet_NaN();
    return false;
  }

  // get depth values from image
  float d_center = meanDepth( ii_depth_map, ii_depth_count, x, y, sp_int);
  float d_xp = meanDepth( ii_depth_map, ii_depth_count, x+sp_int, y, sp_int);
  float d_yp = meanDepth( ii_depth_map, ii_depth_count, x, y+sp_int, sp_int);
  float d_xn = meanDepth( ii_depth_map, ii_depth_count, x-sp_int, y, sp_int);
  float d_yn = meanDepth( ii_depth_map, ii_depth_count, x, y-sp_int, sp_int);

  if ( isnan(d_center) || isnan(d_xp) || isnan(d_yp) || isnan(d_xn) || isnan(d_yn) )
  {
    return false;
  }

  float dxx = d_xp - 2*d_center + d_xn;
  float dyy = d_yp - 2*d_center + d_yn;

  const float cMaxCurvature = 2.0f;
  // test for local planarity
  // TODO note: this does not check for the case of a saddle
  if ( std::abs(dxx + dyy) > 2.0f*cMaxCurvature )
  {
    return false;
  }

// depth gradient between (x+sp) and (x-sp)
  grad[0] = (d_xp - d_xn)*0.5*sp/float(sp_int);
  grad[1] = (d_yp - d_yn)*0.5*sp/float(sp_int);
  return true;
}

// sp : pixel scale
// sw : world scale
inline bool getAffine(
    const Vec2f& grad,
    int x, int y,
    float sp, float sw,
    float &angle, float &major, float &minor,
    Point3f& normal )
{
  // if the gradient is 0, make circle
  if ( grad[0] == 0 && grad[1] == 0 )
  {
    major = minor = sp;
    angle = 0;
    return true;
  }

  // gradient, normalized to length=1

  // compute the minor axis length
  float normal_length_inv = fastInverseSqrt( (grad[0]*grad[0] + grad[1]*grad[1]) / (sw*sw) + 1.0f );
  minor = sp * normal_length_inv;
  // major axis is easy
  major = sp;
  // compute angle
  angle = std::atan2( grad[0], -grad[1] );

  normal.x = grad[0] / sw;
  normal.y = grad[1] / sw;
  normal.z = -1.0f;
  normal = normal * normal_length_inv;

  return true;
}

/** Computes A*x^2 + B*x*y + C*y^2 form of ellipse from angle and major/minor axis length */
inline void ellipseParameters(float angle, float major, float minor, float& A, float& B, float& C)
{
  float ax = std::cos(angle);
  float ay = std::sin(angle);
  float bx = -ay;
  float by = ax;

  float a2 = major * major;
  float b2 = minor * minor;

  A = ax*ax / a2 + bx*bx / b2;

  B = 2.0f * (ax*ay / a2 + bx*by / b2);

  C = ay*ay / a2 + by*by / b2;

}

/** Checks if a point (x,y) is contained in an ellipse of form A*x^2 + B*x*y + C*y^2 */
inline bool ellipseContains(float x, float y, float A, float B, float C)
{
  return A*x*x + B*x*y + C*y*y <= 1.0f;
}

}

#endif
