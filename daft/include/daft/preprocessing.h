/*
 * Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>

#include <math.h>

#ifndef __DAFT_PREPROCESSING_H__
#define __DAFT_PREPROCESSING_H__

namespace cv
{
namespace daft
{

inline float getMeanDx( const cv::Mat1f &depth_map, int y, int x1, int x2 )
{
  x1 = std::min( std::max( 0, x1 ), depth_map.cols-1 );
  x2 = std::min( std::max( 0, x2 ), depth_map.cols-1 );

  float dx_sum=0;
  float dx_num=0;

  for ( int x = x1; x<x2; x++ )
  {
    if ( !isnan( depth_map(y,x) ) && !isnan( depth_map(y,x+1) ) )
    {
      dx_sum += depth_map(y,x+1) - depth_map(y,x);
      dx_num++;
    }
  }
  return dx_sum / dx_num;
}

template< int MaxGapSize >
void closeGapsX( cv::Mat1f &depth_map_out, cv::Mat1f &inter_x, cv::Mat1i &inter_x_dist )
{
  for ( int y=0; y<depth_map_out.rows; y++ )
  {
    int x_right=0;
    int gap_size = 0;

    for ( ; x_right<depth_map_out.cols; x_right++ )
    {
      if ( x_right<depth_map_out.cols-1 && isnan( depth_map_out[y][x_right] ) )
      {
        gap_size++;
        inter_x[y][x_right] = std::numeric_limits<float>::quiet_NaN();
        inter_x_dist[y][x_right] = 1;
      }
      else
      {
        int x_left = x_right-gap_size-1;

        float left_val = depth_map_out[y][x_left];
        float right_val = depth_map_out[y][x_right];

        const int delta_offs = 30;

        if ( isnan(right_val) || left_val > right_val )
        {
          float dx = getMeanDx( depth_map_out, y, x_left-delta_offs, x_left );

          for ( int delta = 1; delta<=gap_size && delta <= MaxGapSize; delta++ )
          {
            inter_x[y][x_left + delta] = left_val + delta * dx;
            inter_x_dist[y][x_left + delta] = delta;
          }
        }
        else if ( isnan(left_val) || left_val <= right_val )
        {
          float dx = getMeanDx( depth_map_out, y, x_right, x_right+delta_offs);

          for ( int delta = 1; delta<=gap_size && delta <= MaxGapSize; delta++ )
          {
            inter_x[y][x_right - delta] = right_val - delta * dx;
            inter_x_dist[y][x_right - delta] = delta;
          }
        }
        gap_size = 0;
      }
    }
  }
}

template< int MaxGapSize >
void closeGaps( const cv::Mat &depth_map_in, cv::Mat1f &depth_map_out, float max_depth_delta )
{
  if ( depth_map_in.type() == CV_16U )
  {
    depth_map_in.convertTo( depth_map_out, CV_32F, 0.001, 0.0 );
  }
  else
  {
    depth_map_out = depth_map_in.clone();
  }

  cv::Mat1f inter_x(depth_map_in.rows, depth_map_in.cols);
  cv::Mat1i inter_x_dist(depth_map_in.rows, depth_map_in.cols);

  closeGapsX<MaxGapSize>( depth_map_out, inter_x, inter_x_dist );

  cv::Mat1f inter_y(depth_map_in.cols, depth_map_in.rows);
  cv::Mat1i inter_y_dist(depth_map_in.cols, depth_map_in.rows);

  cv::Mat1f depth_map_out_t = depth_map_out.t();
  closeGapsX<MaxGapSize>( depth_map_out_t, inter_y, inter_y_dist );

  inter_y = inter_y.t();
  inter_y_dist = inter_y_dist.t();

  for ( int y=0; y<depth_map_out.rows; y++ )
  {
    for ( int x=0; x<depth_map_out.cols; x++ )
    {
      if ( isnan( depth_map_out[y][x] ) )
      {
        if ( isnan( inter_x[y][x] ) )
        {
          depth_map_out[y][x] = inter_y[y][x];
        }
        else if ( isnan( inter_y[y][x] ) )
        {
          depth_map_out[y][x] = inter_x[y][x];
        }
        else
        {
          float wx = MaxGapSize - inter_x_dist[y][x];
          float wy = MaxGapSize - inter_y_dist[y][x];

          depth_map_out[y][x] = ( wx * inter_x[y][x] + wy * inter_y[y][x] ) / ( wx+wy );
        }
      }
    }
  }

/*
  cv::imshow( "inter_x", inter_x );
  cv::imshow( "inter_y", inter_y );
*/
}

}
}

#endif
