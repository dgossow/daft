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

#include "keypoint3d.h"
#include "math_stuff.h"

namespace cv
{

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size at (x,y) will be scale_map(x,y) * base_scale.
 The template argument specifies the filter kernel, which takes as
 arguments the integral image, x, y, and the scaling factor.
 * @param ii            The Integral Image
 * @param scale_map     The scale map (scale multiplier per pixel)
 * @param base_scale    The global scale multiplier
 * @param min_px_scale  Minimal scale in pixels
 * @param max_px_scale  Maximal scale in pixels
 * @param img_out       The output image
 */
template <float (*F)(const Mat1d&, int, int, int)>
void convolve( const cv::Mat1d &ii,
    const cv::Mat1f &scale_map,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
    cv::Mat1f &img_out );

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size and shape will be a local affine transformation.
 The template argument specifies the filter kernel.
 * @param ii            The Integral Image
 * @param scale_map     The scale map (scale multiplier per pixel)
 * @param depth_map     The depth map (im meters)
 * @param camera_matrix Camera intrinsics
 * @param base_scale    The global scale multiplier
 * @param min_px_scale  Minimal scale in pixels
 * @param max_px_scale  Maximal scale in pixels
 * @param img_out       The output image
 */
template <float (*F)(const Mat1d&, const cv::Mat1f&, const cv::Matx33f&, int, int, float, float)>
void convolveAffine( const cv::Mat1d &ii,
    const cv::Mat1f &scale_map,
    const cv::Mat1f &depth_map,
    const cv::Matx33f& camera_matrix,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
    cv::Mat1f &img_out );

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
     std::vector< KeyPoint3D >& kp );

void findMaximaAffine(
    const cv::Mat1d &img,  const cv::Mat1d &scale_map,
    const cv::Mat1d &ii,  const cv::Mat1f &depth_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp );

void findMaximaMipMap( const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp );

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
void filterKpKernel( const cv::Mat1d& ii,
                      double thresh,
                      std::vector< KeyPoint3D >& kp );

/*!
 Check how strong the maximum is in its local neighbourhood
 @param[in]     response   The original kernel response image
 @param[in]     center_fac If this is lower, less keypoints get accepted (range: 0..1)
 @param[in,out] kp         The keypoints (input & output)
 */
void filterKpNeighbours( const cv::Mat1d& response,
    double center_fac,
    std::vector< KeyPoint3D >& kp );


// ----------------------------------------------------
// -- Implementation ----------------------------------
// ----------------------------------------------------

template <float (*F)(const Mat1d&, int, int, int)>
inline float interpolateKernel( const cv::Mat1d &ii,
    int x, int y, float s )
{
  int s_floor = s;
  float t = s - s_floor;
  return (1.0-t) * F( ii, x, y, s_floor ) + t * F( ii, x, y, s_floor+1 );
}
template <float (*F)(const Mat1d &ii, int x, int y, float major, float minor, float angle)>
inline float interpolateKernelAffine( const cv::Mat1d &ii,
    int x, int y, float major, float minor, float angle )
{
  int s_floor = major;
  float t = major - s_floor;
  return (1.0-t) * F( ii, x, y, s_floor, minor, angle ) + t * F( ii, x, y, s_floor+1, minor, angle );
}


template <float (*F)(const Mat1d&, int, int, int)>
void convolve( const cv::Mat1d &ii,
    const cv::Mat1f &scale_map,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
    cv::Mat1f &img_out )
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  img_out.create( ii.rows-1, ii.cols-1 );
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      float s = scale_map[y][x] * base_scale;
      if ( s < min_px_scale || s > max_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      // compute filter response with linear interpolation
      img_out(y,x) = interpolateKernel<F>( ii, x, y, s );

      /*
      int s_round = int ( s + 0.5f );
      img_out(y,x) = F( ii, x, y, s_round );
      */
    }
  }
}

template <float (*F)(const Mat1d&, const cv::Mat1f&, const cv::Matx33f&, int, int, float, float)>
void convolveAffine( const cv::Mat1d &ii,
    const cv::Mat1f &scale_map,
    const cv::Mat1f &depth_map,
    const cv::Matx33f& camera_matrix,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
    cv::Mat1f &img_out )
{
  img_out.create( ii.rows-1, ii.cols-1 );
  float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      float s = scale_map[y][x] * base_scale;
      if ( s < min_px_scale || s > max_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      img_out(y,x) = F( ii, depth_map, camera_matrix, x, y, s, base_scale );
    }
  }
}

template <float (*F)(const Mat1d &ii, int x, int y, float major, float minor, float angle)>
void convolveAffine2( const cv::Mat1d &ii,
    const cv::Mat1f &scale_map,
    const cv::Mat1f &depth_map,
    const cv::Matx33f& camera_matrix,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
    cv::Mat1f &img_out )
{
  img_out.create( ii.rows-1, ii.cols-1 );
  static const float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      float s = scale_map[y][x] * base_scale;
      if ( s < min_px_scale || s > max_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      float angle, major, minor;
      Point3f normal;
      bool ok = getAffine(ii, depth_map,
          x, y, s, base_scale,
          angle, major, minor, normal);
      // break if gradient can not be computed
      // or minor axis too small
      if(!ok || minor < min_px_scale) {
        img_out(y,x) = nan;
        continue;
      }

      img_out(y,x) = interpolateKernelAffine<F>( ii, x, y, major, minor, angle );
      //img_out(y,x) = F( ii, x, y, major, minor, angle );
    }
  }
}


template <float (*F)(const Mat1d&, int, int, int)>
void filterKpKernel( const cv::Mat1d& ii,
    double thresh,
    std::vector< KeyPoint3D >& kp )
{
  std::vector< KeyPoint3D > kp_in = kp;

  kp.clear();
  kp.reserve( kp_in.size() );

  for ( unsigned k=0; k<kp_in.size(); k++ )
  {
    int x = kp_in[k].pt.x;
    int y = kp_in[k].pt.y;

    double s = kp_in[k].size / 4.0;

    float t = s - floor(s);
    float response = (1.0-t) * F( ii, x, y, int(s) ) + t * F( ii, x, y, int(s)+1 );

    if ( response < thresh )
    {
      kp_in[k].response = response;
      kp.push_back( kp_in[k] );
    }
  }
}

template <float (*F)(const Mat1d &ii, int x, int y, float major, float minor, float angle)>
void filterKpKernelAffine( const cv::Mat1d& ii,
    double thresh,
    std::vector< KeyPoint3D >& kp )
{
  std::vector< KeyPoint3D > kp_in = kp;

  kp.clear();
  kp.reserve( kp_in.size() );

  for ( unsigned k=0; k<kp_in.size(); k++ )
  {
    int x = kp_in[k].pt.x;
    int y = kp_in[k].pt.y;

    double major = kp_in[k].affine_major / 4.0;
    double minor = kp_in[k].affine_minor / 4.0;

    float response = interpolateKernelAffine<F>( ii, x, y, major, minor, kp_in[k].affine_angle );

    if ( response < thresh )
    {
      kp_in[k].response = response;
      kp.push_back( kp_in[k] );
    }
  }
}


}

#endif
