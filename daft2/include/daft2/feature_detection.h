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
    const Mat1f& scale_map,
    const Mat4f& affine_map,
    float base_scale,
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
void findExtrema( const Mat1f &img,
     const Mat1f &scale_map,
     double base_scale,
     double min_px_scale,
     double max_px_scale,
     double min_dist,
     double thresh,
     std::vector< KeyPoint3D >& kp );

/*! Like findMaxima, but faster and a little less accurate */
void findMaximaMipMap( const Mat1f &img,
    const Mat1f &scale_map,
    double base_scale,
    double min_px_scale,
    double max_px_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp );

/*! Like findMaxima, but do non-max suppression in affine neighborhood */
void findExtremaAffine(
    const cv::Mat1f &img,
    const Mat1f &scale_map,
    const Mat3f &affine_map,
    double base_scale,
    double min_px_scale,
    double max_px_scale,
    double min_dist,
    double thresh,
    std::vector< KeyPoint3D >& kp );

/*! Reject keypoints with a principal curvature ratio above max_ratio */
void princCurvFilter(
    const Mat1f& response,
    const Mat1f& scale_map,
    const Mat3f& affine_map,
    double max_ratio,
    const std::vector< KeyPoint3D >& kp_in,
    std::vector< KeyPoint3D >& kp_out );

// ----------------------------------------------------
// -- Implementation ----------------------------------
// ----------------------------------------------------

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
    int x, int y,
    float major_len,  float minor_len,
    float major_x, float major_y )>
void convolveAffine( const Mat1d &ii,
    const Mat1f& scale_map,
    const Mat3f& affine_map,
    float base_scale,
    float min_px_scale,
    Mat1f &img_out )
{
  img_out.create( ii.rows-1, ii.cols-1 );

  static const float nan = std::numeric_limits<float>::quiet_NaN();

  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      const float& major_len = base_scale * scale_map[y][x];
      const float& minor_len = major_len * affine_map[y][x][0];
      const float& major_x = affine_map[y][x][1];
      const float& major_y = affine_map[y][x][2];

      if ( isnan( minor_len ) || minor_len < min_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      assert( !isnan( minor_len ) );
      assert( !isnan( major_x ) );
      assert( !isnan( major_y ) );

      img_out(y,x) = F( ii, x, y, major_len, minor_len, major_x, major_y );
    }
  }
}

}
}

#endif
