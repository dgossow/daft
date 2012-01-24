/*
 * Copyright (C) 2011 David Gossow
 */

#ifndef __rgbd_features_keypointdetector_h
#define __rgbd_features_keypointdetector_h

#include <opencv2/features2d/features2d.hpp>

#include <limits>
#include <assert.h>
#include <math.h>

#include <fstream>

namespace cv
{

void findMaxima( const cv::Mat1d &filtered_image,
                 const cv::Mat1d &scale_map,
                 double base_scale,
                 std::vector< KeyPoint >& kp,
                 double threshold = 0.1 );

template <double (*F)(const Mat1d&, int, int, int)>
void filterImage( const cv::Mat1d &ii,
                  const cv::Mat1d &scale_map,
                  double base_scale,
                  cv::Mat1d &filtered_image )
{
  for ( int y = 0; y < ii.rows; y++ )
  {
    for ( int x = 0; x < ii.cols; ++x )
    {
      double s = scale_map[y][x] * base_scale;
      if ( s <= 2.0 )
      {
        filtered_image[y][x] = std::numeric_limits<double>::quiet_NaN();
        continue;
      }

      float t = s - floor(s);
      filtered_image[y][x] = (1.0-t) * F( ii, x, y, int(s) ) + t * F( ii, x, y, int(s)+1 );
    }
  }
}

template <double (*F)(const Mat1d&, int, int, int)>
void filterKeypoints( const cv::Mat1d& ii,
    std::vector< KeyPoint >& kp,
    double threshold = 0.1 )
{
  std::vector< KeyPoint > kp_in = kp;

  kp.clear();
  kp.reserve( kp_in.size() );

#ifdef DEBUG_OUTPUT
  std::fstream fall;
  fall.open( "/tmp/all_resp", std::ios_base::out );
  std::fstream ffiltered;
  ffiltered.open( "/tmp/filtered_resp" );
#endif

  for ( unsigned k=0; k<kp_in.size(); k++ )
  {
    int x = kp_in[k].pt.x;
    int y = kp_in[k].pt.y;

    double s = kp_in[k].size;

    float t = s - floor(s);
    double response = (1.0-t) * F( ii, x, y, int(s) ) + t * F( ii, x, y, int(s)+1 );

    if ( response > threshold )
    {
      kp.push_back( kp_in[k] );
    }

#ifdef DEBUG_OUTPUT
    fall << response << " " << kp_in[k]._score << std::endl;
    if ( response > threshold )
      ffiltered << response << " " << kp_in[k]._score << std::endl;
#endif
  }

#ifdef DEBUG_OUTPUT
  fall.close();
  ffiltered.close();
#endif
}

}

#endif
