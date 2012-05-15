#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "stuff.h"
#include "depth_filter.h"

namespace cv
{
namespace daft2
{

void computeAffineMapFixed(
    const Mat1f &depth_map,
    float sp,
    float f,
    Mat3f& affine_map )
{
  affine_map.create( depth_map.rows, depth_map.cols );

  const float nan = std::numeric_limits<float>::quiet_NaN();
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
      }
      else
      {
        // if the gradient is 0, make circle
        if ( grad[0] == 0 && grad[1] == 0 )
        {
          affine_map[y][x][0] = 1;
          affine_map[y][x][1] = 1.0f;
          affine_map[y][x][2] = 0.0f;
          continue;
        }

        const float grad_len_sqr = grad[0]*grad[0] + grad[1]*grad[1];

        float sw = sp_div_f * depth_map[y][x];

        // major/minor axis lengths
        affine_map[y][x][0] = fastInverseSqrt( grad_len_sqr / (sw*sw) + 1.0f );

        // major axis as unit vector
        const float grad_len_inv = fastInverseSqrt(grad_len_sqr);
        affine_map[y][x][1] = -grad[1] * grad_len_inv;
        affine_map[y][x][2] = grad[0] * grad_len_inv;
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
  if ( affine_map.rows != 0 )
  {
      // we already have an image, no need to compute
      return;
  }

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
      }
      else
      {
        // if the gradient is 0, make circle
        if ( grad[0] == 0 && grad[1] == 0 )
        {
          affine_map[y][x][0] = sp;
          affine_map[y][x][1] = 1.0f;
          affine_map[y][x][2] = 0.0f;
          continue;
        }

        const float grad_len_sqr = grad[0]*grad[0] + grad[1]*grad[1];

        // major/minor axis lengths
        affine_map[y][x][0] = sp * fastInverseSqrt( grad_len_sqr / (sw*sw) + 1.0f );

        // major axis as unit vector
        const float grad_len_inv = fastInverseSqrt(grad_len_sqr);
        affine_map[y][x][1] = -grad[1] * grad_len_inv;
        affine_map[y][x][2] = grad[0] * grad_len_inv;
      }
    }
  }
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
