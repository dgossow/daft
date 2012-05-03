/*
 * Copyright (C) 2011 David Gossow
 */

#ifndef __DAFT2_FEATURE_DETECTION_H__
#define __DAFT2_FEATURE_DETECTION_H__

#include <opencv2/features2d/features2d.hpp>

#include <limits>
#include <assert.h>
#include <math.h>

#include <fstream>

#include <opencv2/features3d/features3d.hpp>
#include "stuff.h"

namespace cv
{
namespace daft2
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
void convolve( const Mat1d &ii,
    const Mat1f &scale_map,
    float base_scale,
    float min_px_scale,
    Mat1f &img_out );

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size and shape will be a local affine transformation.
 The template argument specifies the filter kernel.
 * @param ii            integral image
 * @param scale_map     scale map (scale multiplier per pixel)
 * @param depth_map     depth map [meter]
 * @param sw            world scale [meter]
 * @param min_px_scale  Minimal scale in pixels
 * @param img_out       The output image
 */
template <float (*F)( const Mat1d &ii,
    int x, int y, float sp,
    float sw, float major_x, float major_y,
    float minor_ratio, float min_sp )>
void convolveAffine( const Mat1d &ii,
    const Mat1f &scale_map,
    const Mat3f& affine_map,
    float sw,
    float min_px_scale,
    Mat1f &img_out );

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
void findMaxima( const Mat1d &img,
     const Mat1d &scale_map,
     double base_scale,
     double min_px_scale,
     double max_px_scale,
     double thresh,
     std::vector< KeyPoint3D >& kp );

/*! Like findMaxima, but faster and a little less accurate */
void findMaximaMipMap( const Mat1d &img,
    const Mat1d &scale_map,
    double base_scale,
    double min_px_scale,
    double max_px_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp );

/*! Like findMaxima, but do non-max suppression in affine neighborhood */
void findMaximaAffine(
    const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    const Mat3f &affine_map,
    double base_scale,
    double min_px_scale,
    double max_px_scale,
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
void filterKpKernel( const Mat1d& ii,
                      double thresh,
                      std::vector< KeyPoint3D >& kp );

/*!
 Check how strong the maximum is in its local neighbourhood
 @param[in]     response   The original kernel response image
 @param[in]     center_fac If this is lower, less keypoints get accepted (range: 0..1)
 @param[in,out] kp         The keypoints (input & output)
 */
void filterKpNeighbours( const Mat1d& response,
    double center_fac,
    std::vector< KeyPoint3D >& kp );


// ----------------------------------------------------
// -- Implementation ----------------------------------
// ----------------------------------------------------

template <float (*F)(const Mat1d&, int, int, int)>
inline float interpolateKernel( const Mat1d &ii,
    int x, int y, float s )
{
  int s_floor = s;
  float t = s - s_floor;
  return (1.0-t) * F( ii, x, y, s_floor ) + t * F( ii, x, y, s_floor+1 );
}

template <float (*F)(const Mat1d&, int, int, float)>
void convolve( const Mat1d &ii,
    const Mat1f &scale_map,
    float base_scale,
    float min_px_scale,
    Mat1f &img_out )
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  img_out.create( ii.rows-1, ii.cols-1 );
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      const float s = getScale(scale_map[y][x], base_scale);

      if ( s < min_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      img_out(y,x) = F( ii, x, y, s );
    }
  }
}

template <float (*F)( const Mat1d &ii,
    int x, int y, float sp,
    float sw, float major_x, float major_y,
    float minor_ratio, float min_sp )>
void convolveAffine( const Mat1d &ii,
    const Mat1f &scale_map,
    const Mat3f& affine_map,
    float sw,
    float min_px_scale,
    Mat1f &img_out )
{
  img_out.create( ii.rows-1, ii.cols-1 );

  static const float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      const float sp = getScale(scale_map[y][x], sw);

      if ( isnan( affine_map[y][x][0] ) || sp < min_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      img_out(y,x) = F( ii, x, y, sp, sw, affine_map[y][x][0],
          affine_map[y][x][1], affine_map[y][x][2], min_px_scale );
    }
  }
}


template <float (*F)(const Mat1d&, int, int, float)>
void filterKpKernel( const Mat1d& ii,
    double thresh,
    std::vector< KeyPoint3D >& kp )
{
  std::vector< KeyPoint3D > kp_in = kp;

  kp.clear();
  kp.reserve( kp_in.size() );

  for ( unsigned k=0; k<kp_in.size(); k++ )
  {
    float response = F( ii,
        kp_in[k].pt.x, kp_in[k].pt.y,
        kp_in[k].size / 4.0f);

    if ( response < thresh )
    {
      //kp_in[k].response = response;
      kp.push_back( kp_in[k] );
    }
  }
}

template <float (*F)(const Mat1d &ii, int x, int y, float major, float minor, float angle)>
void filterKpKernelAffine( const Mat1d& ii,
    double thresh,
    std::vector< KeyPoint3D >& kp )
{
  std::vector< KeyPoint3D > kp_in = kp;

  kp.clear();
  kp.reserve( kp_in.size() );

  for ( unsigned k=0; k<kp_in.size(); k++ )
  {
    float response = F( ii,
        kp_in[k].pt.x, kp_in[k].pt.y,
        kp_in[k].affine_major / 4.0, kp_in[k].affine_minor / 4.0, kp_in[k].affine_angle );

    if ( response < thresh )
    {
      //kp_in[k].response = response;
      kp.push_back( kp_in[k] );
    }
  }
}

}
}

#endif
