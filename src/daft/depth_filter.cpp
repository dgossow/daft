#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"
#include "depth_filter.h"

namespace cv
{
namespace daft
{


inline void getAffParams( float sw, Vec2f grad, cv::Vec3f &params )
{
  // if the gradient is 0, make circle
  if ( grad[0] == 0 && grad[1] == 0 )
  {
    params[0] = 1.0f;
    params[1] = 1.0f;
    params[2] = 0.0f;
    return;
  }

  const float grad_len_sqr = grad[0]*grad[0] + grad[1]*grad[1];

  // major/minor axis lengths
  params[0] = fastInverseSqrt( grad_len_sqr / (sw*sw) + 1.0f );

  // major axis as unit vector
  const float grad_len_inv = fastInverseSqrt(grad_len_sqr);
  params[1] = -grad[1] * grad_len_inv;
  params[2] = grad[0] * grad_len_inv;
}

void computeAffineMapFixed(
    const Mat1f &depth_map,
    float sp,
    float f,
    Mat3f& affine_map )
{
  affine_map.create( depth_map.rows, depth_map.cols );

  const float sp_div_f = sp / f;

  for ( int y = 0; y < depth_map.rows; y++ )
  {
    for ( int x = 0; x < depth_map.cols; ++x )
    {
      Vec2f grad;
      if ( isnan(depth_map[y][x]) ||
           !computeGradient( depth_map, x, y, sp, grad ) )
      {
        affine_map[y][x][0] = nan;
        affine_map[y][x][1] = nan;
        affine_map[y][x][2] = nan;
      }
      else
      {
        const float sw = sp_div_f * depth_map[y][x];
        getAffParams( sw, grad, affine_map[y][x] );
      }
    }
  }
}


void computeAffineMap(
    const Mat1f &scale_map,
    const Mat1f &depth_map,
    float sw,
    float min_px_scale,
    Mat3f& affine_map )
{
  affine_map.create( depth_map.rows, depth_map.cols );

  const float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int y = 0; y < depth_map.rows; y++ )
  {
    for ( int x = 0; x < depth_map.cols; ++x )
    {
      Vec2f grad;
      const float sp = getScale(scale_map[y][x], sw);
      if ( isnan(sp) || sp < min_px_scale ||
           !computeGradient( depth_map, x, y, sp, grad ) )
      {
        affine_map[y][x][0] = nan;
        affine_map[y][x][1] = nan;
        affine_map[y][x][2] = nan;
      }
      else
      {
        getAffParams( sw, grad, affine_map[y][x] );
      }
    }
  }
}


inline float meanDepth(const Mat1d &ii_depth_map,
    const cv::Mat_<uint32_t>& ii_depth_count,
    int x, int y, int sp_int )
{
  int x_left = x-sp_int;
  int x_right = x+sp_int;
  int y_top = y-sp_int;
  int y_bottom = y+sp_int;
  if ( x_left < 0 ) x_left = 0;
  if ( y_top < 0 ) y_top = 0;
  if ( x_right >= ii_depth_map.cols ) x_right = ii_depth_map.cols-1;
  if ( y_bottom >= ii_depth_map.rows ) y_bottom = ii_depth_map.rows-1;
  float nump = float(integrate( ii_depth_count, x_left, y_top, x_right, y_bottom ));
  if ( nump == 0 )
  {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return integrate( ii_depth_map, x_left, y_top, x_right, y_bottom ) / nump;
}

void smoothDepth( const Mat1f &scale_map,
    const Mat1d &ii_depth_map,
    const Mat_<uint32_t>& ii_depth_count,
    float base_scale,
    Mat1f &depth_out )
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  depth_out.create( ii_depth_map.rows-1, ii_depth_map.cols-1 );
  for ( int y = 0; y < ii_depth_map.rows-1; y++ )
  {
    for ( int x = 0; x < ii_depth_map.cols-1; ++x )
    {
      float s = scale_map[y][x] * base_scale + 0.5f;

      if ( isnan(s) )
      {
        depth_out(y,x) = nan;
        continue;
      }

      if ( s < 5 ) s=5;

      const int s_floor = s;
      const float t = s - s_floor;

      if ( !checkBounds( ii_depth_map, x, y, 1 ) )
      {
        depth_out(y,x) = nan;
        continue;
      }

      depth_out(y,x) = (1.0-t) * meanDepth( ii_depth_map, ii_depth_count, x, y, s_floor )
          + t * meanDepth( ii_depth_map, ii_depth_count, x, y, s_floor+1 );
    }
  }
}

}
}
