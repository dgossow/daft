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

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size at (x,y) will be scale_map(x,y) * base_scale.
 The template argument specifies the filter kernel, which takes as
 arguments the integral image, x, y, and the scaling factor.
 @tparam     F          The kernel function f(ii,x,y,s)
 @param[in]  ii         The Integral Image
 @param[in]  scale_map  The scale map (scale multiplier per pixel)
 @param[in]  base_scale The global scale multiplier
 @param[out] img_out    The output image
 */
template <double (*F)(const Mat1d&, int, int, int)>
void filterImage( const cv::Mat1d &ii,
                  const cv::Mat1d &scale_map,
                  double base_scale,
                  cv::Mat1d &img_out );

/*!
 Find the local maxima in the given image with a minimal value of thresh,
 The width & height of the local neighbourhood searched is
 scale_map(x,y) * base_scale.
 @param[in]  img        The input image
 @param[in]  scale_map  The scale map (scale multiplier per pixel)
 @param[in]  base_scale The global scale multiplier
 @param[in]  thresh     Minimum threshold for local maxima
 @param[out] kp         The keypoints (input & output)
 */
void findMaxima( const cv::Mat1d &img,
     const cv::Mat1d &scale_map,
     double base_scale,
     double thresh,
     std::vector< KeyPoint >& kp );

void findMaximaMipMap( const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint >& kp );

/*!
 Compute the kernel response for each keypoint and reject those
 with a response below the threshold.
 The kernel size at (x,y) will be scale_map(x,y) * base_scale.
 The template argument specifies the filter kernel, which takes as
 arguments the integral image, x, y, and the scaling factor.
 @tparam        F       The kernel function f(ii,x,y,s)
 @param[in]     ii      The integral image
 @param[in]     thresh  Keypoint with a lower kernel response below this will be
 @param[in,out] kp      The keypoints (input & output)
 */
template <double (*F)(const Mat1d&, int, int, int)>
void filterKeypoints( const cv::Mat1d& ii,
                      double thresh,
                      std::vector< KeyPoint >& kp );



// ----------------------------------------------------
// -- Implementation ----------------------------------
// ----------------------------------------------------

template <double (*F)(const Mat1d&, int, int, int)>
void filterImage( const cv::Mat1d &ii,
                  const cv::Mat1d &scale_map,
                  double base_scale,
                  cv::Mat1d &img_out )
{
  img_out.create( ii.rows-1, ii.cols-1 );
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      double s = scale_map[y][x] * base_scale;
      if ( s <= 2.0 )
      {
        img_out(y,x) = 0;//std::numeric_limits<double>::quiet_NaN();
        continue;
      }

      int s_floor = floor(s);
      float t = s - s_floor;
      img_out(y,x) = (1.0-t) * F( ii, x, y, s_floor ) + t * F( ii, x, y, s_floor+1 );
      img_out(y,x) *= 255;
    }
  }
}

template <double (*F)(const Mat1d&, int, int, int)>
void filterKeypoints( const cv::Mat1d& ii,
                      double thresh,
                      std::vector< KeyPoint >& kp )
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

    if ( response > thresh )
    {
      kp.push_back( kp_in[k] );
    }

#ifdef DEBUG_OUTPUT
    fall << response << " " << kp_in[k]._score << std::endl;
    if ( response > thresh )
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
