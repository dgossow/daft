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

#include "keypoint3d.h"
#include "stuff.h"

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
void convolve( const Mat1d &ii,
    const Mat1f &scale_map,
    float base_scale,
    float min_px_scale,
    float max_px_scale,
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
     double thresh,
     std::vector< KeyPoint3D >& kp );

void findMaximaAffine(
    const cv::Mat1d &img,  const cv::Mat1d &scale_map,
    const Mat2f &grad_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp );

void findMaximaMipMap( const Mat1d &img,
    const Mat1d &scale_map,
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

void diff( const Mat1f& l1, const Mat1f& l2, Mat1f& out );


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
    float max_px_scale,
    Mat1f &img_out )
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

      img_out(y,x) = F( ii, x, y, s );
    }
  }
}

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size and shape will be a local affine transformation.
 The template argument specifies the filter kernel.
 * @param ii            The Integral Image
 * @param scale_map     The scale map (scale multiplier per pixel)
 * @param ii_depth_map  The integral depth map (im meters)
 * @param ii_depth_countIntegral of the number of valid depth pixels
 * @param camera_matrix Camera intrinsics
 * @param base_scale    The global scale multiplier
 * @param min_px_scale  Minimal scale in pixels
 * @param max_px_scale  Maximal scale in pixels
 * @param img_out       The output image
 */
template <float (*F)( const Mat1d &ii, Vec2f &grad,
    int x, int y, float sp, float sw, float min_sp )>
void convolveAffine( const Mat1d &ii,
    const Mat1f &scale_map,
    const Mat1d &ii_depth_map, const Mat_<uint64_t>& ii_depth_count,
    float sw,
    float min_px_scale,
    float max_px_scale,
    Mat1f &img_out,
    Mat2f& depth_grad )
{
  img_out.create( ii.rows-1, ii.cols-1 );

  // determine if we need to compute the depth gradient
  const bool compute_grad = ( depth_grad.rows != ii.rows-1 || depth_grad.cols != ii.cols-1 );
  if ( compute_grad ) {
    depth_grad.create( ii.rows-1, ii.cols-1 );
  }

  static const float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      const float sp = scale_map[y][x] * sw;
      if ( compute_grad )
      {
        computeGradient( ii_depth_map, ii_depth_count, x, y, sp, depth_grad[y][x] );
      }

      if ( std::isnan( depth_grad[y][x][0] ) || sp < min_px_scale || sp > max_px_scale )
      {
        img_out(y,x) = nan;
        continue;
      }

      img_out(y,x) = F( ii, depth_grad[y][x], x, y, sp, sw, min_px_scale );
    }
  }
}

/*!
 Compute the kernel response for every pixel of the given image.
 The kernel size and shape will be a local affine transformation.
 The template argument specifies the filter kernel.
 * @param ii            The Integral Image
 * @param scale_map     The scale map (scale multiplier per pixel)
 * @param ii_depth_map  The integral depth map (im meters)
 * @param ii_depth_countIntegral of the number of valid depth pixels
 * @param camera_matrix Camera intrinsics
 * @param base_scale    The global scale multiplier
 * @param min_px_scale  Minimal scale in pixels
 * @param max_px_scale  Maximal scale in pixels
 * @param img_out       The output image
 */
template < float (*Fx)( const Mat1d &ii, Vec2f &grad,
    int x, int y, float sp, float sw, float min_sp ),
    float (*Fy)( const Mat1f &ii_y, const Mat1f &ii_y_count, Vec2f &grad,
        int x, int y, float sp, float sw, float min_sp ) >
void convolveAffineSep( const Mat1d &ii,
    const Mat1f &scale_map,
    const Mat1d &ii_depth_map, const Mat_<uint64_t>& ii_depth_count,
    float sw,
    float min_px_scale,
    float max_px_scale,
    Mat1f &img_out,
    Mat2f& depth_grad )
{
  img_out.create( ii.rows-1, ii.cols-1 );

  Mat1f ii_y( ii.rows, ii.cols );
  Mat1f ii_y_count( ii.rows, ii.cols );

  // determine if we need to compute the depth gradient
  const bool compute_grad = ( depth_grad.rows != ii.rows-1 || depth_grad.cols != ii.cols-1 );
  if ( compute_grad ) {
    depth_grad.create( ii.rows-1, ii.cols-1 );
  }

  static const float nan = std::numeric_limits<float>::quiet_NaN();
  for ( int x = 0; x < ii.cols-1; ++x )
  {
    ii_y[0][x] = 0;
    ii_y_count[0][x] = 0;
  }
  // convolute in x direction and integrate in y direction
  for ( int y = 1; y < ii.rows; y++ )
  {
    for ( int x = 0; x < ii.cols; ++x )
    {
      const float sp = scale_map[y][x] * sw;
      if ( compute_grad )
      {
        computeGradient( ii_depth_map, ii_depth_count, x, y, sp, depth_grad[y][x] );
      }

      if ( std::isnan( depth_grad[y][x][0] ) || sp < min_px_scale || sp > max_px_scale )
      {
        ii_y(y,x) = ii_y(y-1,x);
        ii_y_count(y,x) = ii_y_count(y-1,x);
        continue;
      }

      ii_y(y,x) = ii_y(y-1,x) + Fx( ii, depth_grad[y][x], x, y, sp, sw, min_px_scale );
      ii_y_count(y,x) = ii_y_count(y-1,x) + 1.0f;
    }
  }
  // convolute in major axis direction using y integral
  for ( int y = 0; y < ii.rows-1; y++ )
  {
    for ( int x = 0; x < ii.cols-1; ++x )
    {
      float sp = scale_map[y][x] * sw;
      img_out(y,x) = ii_y[y+1][x] - ii_y[y][x];//Fy( ii_y, ii_y_count, depth_grad[y][x], x, y, sp, sw, min_px_scale );
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
      kp_in[k].response = response;
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
      kp_in[k].response = response;
      kp.push_back( kp_in[k] );
    }
  }
}


}

#endif
