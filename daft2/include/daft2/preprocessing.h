/*
 * Copyright (C) 2011 David Gossow
*/

#include <opencv2/core/core.hpp>

#include <math.h>

#ifndef __DAFT2_PREPROCESSING_H__
#define __DAFT2_PREPROCESSING_H__

namespace cv
{
namespace daft2
{

template< int MaxGapSize >
void improveDepthMap( const cv::Mat &depth_map_in, cv::Mat1f &depth_map_out, float max_depth_delta )
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
  cv::Mat1f inter_y(depth_map_in.rows, depth_map_in.cols);

  cv::Mat1i inter_x_dist(depth_map_in.rows, depth_map_in.cols);
  cv::Mat1i inter_y_dist(depth_map_in.rows, depth_map_in.cols);

  for ( int y=0; y<depth_map_out.rows; y++ )
  {
    int x=0;
    while( x<depth_map_out.cols && isnan(depth_map_out[y][x]))
    {
      inter_x[y][x] = std::numeric_limits<float>::quiet_NaN();
      x++;
    }

    int gap_size = 0;

    for ( ; x<depth_map_out.cols; x++ )
    {
      if ( isnan( depth_map_out[y][x] ) )
      {
        gap_size++;
        inter_x[y][x] = std::numeric_limits<float>::quiet_NaN();
        inter_x_dist[y][x] = 1;
      }
      else
      {
        //inter_y[y][x] = std::numeric_limits<float>::quiet_NaN();
        if ( gap_size < MaxGapSize )
        {
          int x_left = x-gap_size-1;

          float left_val = depth_map_out[y][x_left];
          float right_val = depth_map_out[y][x];

          float depth_delta = std::abs(left_val - right_val) / ( left_val + right_val );

          if (depth_delta < max_depth_delta )
          {

            for ( int delta = 1; delta<=gap_size; delta++ )
            {
              float t = float(delta) / float(gap_size+2);
              inter_x[y][x_left + delta] = (1.0-t) * left_val + t * right_val;
              inter_x_dist[y][x] = gap_size;
            }
          }
        }
        gap_size = 0;
      }
    }
  }

  for ( int x=0; x<depth_map_out.cols; x++ )
  {
    int y=0;
    while( y<depth_map_out.rows && isnan(depth_map_out[y][x]) )
    {
      inter_y[y][x] = std::numeric_limits<float>::quiet_NaN();
      inter_y_dist[y][x] = 1;
      y++;
    }

    int gap_size = 0;

    for ( ; y<depth_map_out.rows; y++ )
    {
      if ( isnan( depth_map_out[y][x] ) )
      {
        gap_size++;
        inter_y[y][x] = std::numeric_limits<float>::quiet_NaN();
      }
      else
      {
        //inter_y[y][x] = std::numeric_limits<float>::quiet_NaN();
        if ( gap_size < MaxGapSize )
        {
          int y_top = y-gap_size-1;

          float top_val = depth_map_out[y_top][x];
          float bottom_val = depth_map_out[y][x];

          float depth_delta = std::abs(top_val - bottom_val) / ( top_val + bottom_val );

          if (depth_delta < max_depth_delta )
          {
            for ( int delta = 1; delta<=gap_size; delta++ )
            {
              float t = float(delta) / float(gap_size+2);
              inter_y[y_top + delta][x] = (1.0-t) * top_val + t * bottom_val;
              inter_y_dist[y][x] = gap_size;
            }
          }
        }
        gap_size = 0;
      }
    }
  }

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
