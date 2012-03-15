/*
* Copyright (C) 2011 David Gossow
*/

#ifndef rgbd_features_descriptor_computation_h_
#define rgbd_features_descriptor_computation_h_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "filter_kernels.h"

#include <math.h>
#include <list>

namespace cv
{
namespace daft
{

void imshow2( std::string win_title, cv::Mat img, int size = 256 )
{
  cv::Mat img2;
  cv::resize( img, img2, Size( size, size), 0, 0, INTER_NEAREST );
  imshow( win_title, img2 );
}


template< int PatchSize >
inline void getPatch( const cv::Mat1d& ii, const KeyPoint3D& kp, float world_radius, cv::Mat1f& patch )
{
  patch.create( PatchSize, PatchSize );

  //const Point2f& pt
  float relative_scale = world_radius / ( kp.world_size * 0.5 * float(PatchSize) );
  float angle = kp.affine_angle;
  float major = kp.affine_major;
  float minor = kp.affine_minor;

  int s_round = int( 0.25 * relative_scale * sqrt(major*minor) + 0.5 );
  if ( s_round < 1 ) s_round = 1;

  Matx22f rot_mat;
  rot_mat(0,0) = cos(angle);
  rot_mat(1,0) = sin(angle);
  rot_mat(0,1) = -rot_mat( 1, 0 );
  rot_mat(1,1) = rot_mat( 0, 0 );

  major *= relative_scale;
  minor *= relative_scale;

  //float len_major = sqrt( affine_mat(0,0)*affine_mat(0,0) + affine_mat(1,0)*affine_mat(1,0) );

  for ( int u = 0; u<PatchSize; u++ )
  {
    for ( int v = 0; v<PatchSize; v++ )
    {
      Point2f uv( float(v-PatchSize/2) , float(u-PatchSize/2) );
      uv = rot_mat.t() * uv;
      uv.x*=major;
      uv.y*=minor;
      Point2f pixel = rot_mat * uv + kp.pt;
      if ( checkBounds( ii, pixel.x, pixel.y, s_round ) )
      {
        patch[u][v] = iiMean( ii, pixel.x, pixel.y, s_round );
      }
    }
  }
}

struct PtInfo {
  Point3f pos;
  float intensity;
  Point2f grad;
  float weight;
};
struct PtGradient
{
  float grad_ori;
  float grad_mag;
  inline bool operator< ( const PtGradient& other ) const { return grad_ori < other.grad_ori; }
};


inline float dominantOri( std::vector< PtGradient >& gradients )
{
  // sort by orientation
  sort( gradients.begin(), gradients.end() );

  // find dominant orientation

  //estimate orientation using a sliding window of PI/3
  PtGradient grad_max = gradients[0];
  double grad_mag_sum = gradients[0].grad_mag;
  double grad_ori_sum = gradients[0].grad_ori * grad_mag_sum;

  size_t window_begin = 0;
  size_t window_end = 0;

  float ori_offset = 0;

  std::list< std::pair<float, float> > hist_values;
  float max_mag = 0;

  while ( window_begin < gradients.size() )
  {
      float window_size = ( gradients[window_end].grad_ori + ori_offset ) - gradients[window_begin].grad_ori;
      if ( window_size < M_PI / 3 )
      {
          //found new max.
          if ( grad_mag_sum > grad_max.grad_mag )
          {
              grad_max.grad_mag = grad_mag_sum;
              grad_max.grad_ori = grad_ori_sum;
          }
          window_end++;
          if ( window_end >= gradients.size() )
          {
              window_end = 0;
              ori_offset += 2 * M_PI;
          }
          grad_mag_sum += gradients[window_end].grad_mag;
          grad_ori_sum += gradients[window_end].grad_mag * ( gradients[window_end].grad_ori + ori_offset );

          hist_values.push_back( std::pair<float,float>( grad_ori_sum/grad_mag_sum , grad_mag_sum ) );
          max_mag = std::max( max_mag, gradients[window_end].grad_mag );
      }
      else
      {
          grad_mag_sum -= gradients[window_begin].grad_mag;
          grad_ori_sum -= gradients[window_begin].grad_mag * gradients[window_begin].grad_ori;
          window_begin++;
      }
  }

  float dominant_ori = grad_max.grad_ori / grad_max.grad_mag;

  // debug visualization
  cv::Mat3b hist_img( 220, 220, 0.0f );
  for ( std::list< std::pair<float, float> >::iterator it = hist_values.begin(); it != hist_values.end(); ++it )
  {
    float ori = it->first;
    float mag = it->second / grad_max.grad_mag*100;

    cv::line( hist_img, Point(110,110), Point(110+cos(ori)*mag, 110+sin(ori)*mag ), Scalar(0,255,0), 1, 16 );
  }
  cv::line( hist_img, Point(110,110), Point(110+cos(dominant_ori)*100, 110+sin(dominant_ori)*100 ), Scalar(0,0,255), 1, 16 );
  imshow( "ori hist", hist_img );

  // debug visualization
  cv::Mat3b grad_img( 440, 440, 0.0f );
  for ( std::vector< PtGradient >::iterator it = gradients.begin(); it != gradients.end(); ++it )
  {
    float ori = it->grad_ori;
    float mag = it->grad_mag/max_mag*200;
    grad_img(220+sin(ori)*mag,220+cos(ori)*mag) = Vec3b(0,255,0);
  }
  cv::line( grad_img, Point(220,220), Point(220+cos(dominant_ori)*200, 220+sin(dominant_ori)*200 ), Scalar(0,0,255), 1, 16 );
  imshow( "ori", grad_img );

  return dominant_ori;
}


template< int PatchSize >
inline void getPatch2( const cv::Mat1d& ii, const cv::Mat1f depth_map, cv::Matx33f& K,
    const KeyPoint3D& kp, float world_radius, cv::Mat1f& patch, cv::Mat& img )
{
  patch.create( PatchSize, PatchSize );
  float relative_scale = world_radius / ( kp.world_size * 0.5 * float(PatchSize) );

  for ( int u = 0; u<PatchSize; u++ )
  {
    for ( int v = 0; v<PatchSize; v++ )
    {
      patch[u][v] = 0;
    }
  }

  float f_inv = 1.0 / K(0,0);
  float cx = K(0,2);
  float cy = K(1,2);

  cv::Matx33f M;

  Point3f affine_major_axis( cos(kp.affine_angle), sin(kp.affine_angle), 0 );

  float affine_ratio = kp.affine_major / kp.affine_minor;

  cv::Matx22f affine_mat( affine_major_axis.x, affine_major_axis.y,
      - affine_major_axis.y / affine_ratio, affine_major_axis.x / affine_ratio );

  Point3f normal = kp.normal;
  Point3f v1 = normal.cross( affine_major_axis );
  v1 = v1  * fastInverseLen( v1 );
  Point3f v2 = v1.cross( normal );

  M( 0,0 ) = v1.x;
  M( 1,0 ) = v1.y;
  M( 2,0 ) = v1.z;

  M( 0,1 ) = v2.x;
  M( 1,1 ) = v2.y;
  M( 2,1 ) = v2.z;

  M( 0,2 ) = normal.x;
  M( 1,2 ) = normal.y;
  M( 2,2 ) = normal.z;

  /*
  Vec2f depth_grad;
  computeGradient( depth_map, kp.pt.x, kp.pt.y, kp.size/2, kp.world_size/2, depth_grad );
  // compute dx per pixel
  depth_grad *= 2/kp.world_size;

  float grad_norm_x = 1.0 / sqrt( depth_grad[0]*depth_grad[0] + 1 );
  float grad_norm_y = 1.0 / sqrt( depth_grad[1]*depth_grad[1] + 1 );

  std::cout << " grad_norm_x " << grad_norm_x << std::endl;
  std::cout << " grad_norm_y " << grad_norm_y << std::endl;
  */

  cv::Matx33f M_inv = M.t() * ( 1.0 / world_radius * ((float)PatchSize / 2.0) );

  {
    Point3f pn3d = kp.pt3d + (normal*kp.world_size);
    Point3f pv13d = kp.pt3d + (v1*kp.world_size);
    Point3f pv23d = kp.pt3d + (v2*kp.world_size);

    Point2f p,pn,pv1,pv2;

    getPt2d( pn3d, (float)K(0,0), cx, cy, pn );
    getPt2d( pv13d, (float)K(0,0), cx, cy, pv1 );
    getPt2d( pv23d, (float)K(0,0), cx, cy, pv2 );
    getPt2d( kp.pt3d, (float)K(0,0), cx, cy, p );

    cv::line( img, p, pv1, cv::Scalar( 0,0,255 ), 1, 16 );
    cv::line( img, p, pv2, cv::Scalar( 0,255,0 ), 1, 16 );
    cv::line( img, p, pn, cv::Scalar( 255,0,0 ), 1, 16 );
  }

#if 0
  for ( int u = 0; u<3; u++ )
  {
    for ( int v = 0; v<3; v++ )
    {
      std::cout << M_inv(u,v) << " ";
    }
    std::cout << std::endl;
  }

  std::cout << std::endl;
  //usleep( 3000000 );
#endif

  float s = 0.25 * relative_scale * sqrt(kp.affine_major*kp.affine_minor);
  if ( s < 1 ) s = 1;

  int s_ceil = std::ceil( s );

  int win_size = std::ceil( kp.size * 0.25 * float(PatchSize) * relative_scale );

  int step_size = std::ceil( kp.affine_minor * relative_scale );

  // for debug
  step_size /= 2;

  if ( step_size < 1 ) step_size = 1;

  int start_x = kp.pt.x-win_size;
  int end_x = kp.pt.x+win_size;
  int start_y = kp.pt.y-win_size;
  int end_y = kp.pt.y+win_size;

  if ( start_x < s_ceil*2 ) start_x = s_ceil*2;
  if ( start_y < s_ceil*2 ) start_y = s_ceil*2;
  if ( end_x > ii.cols-s_ceil*2-1 ) end_x = ii.cols-s_ceil*2-1;
  if ( end_y > ii.rows-s_ceil*2-1 ) end_y = ii.rows-s_ceil*2-1;

  cv::rectangle( img, Point(start_x,start_y), Point(end_x,end_y), cv::Scalar(0,0,0) );

  std::vector< PtInfo > pts;
  std::vector< PtGradient > gradients;

  pts.reserve( (end_x-start_x)*(end_y-start_y) );
  gradients.reserve( pts.size() );

  float depth_norm = 1.0 / kp.pt3d.z;

  for ( int y=start_y; y<end_y; y+=step_size )
  {
    for ( int x=start_x; x<=end_x; x+=step_size )
    {
      if ( finite(depth_map[y][x]) )
      {
        // get 3d point
        Point3f pt3d;
        getPt3d( f_inv, cx, cy, x, y, depth_map[y][x], pt3d );

        // compute relative translation
        Point3f pt3d_local = M_inv * (pt3d - kp.pt3d);

        float dist_2 = (pt3d_local.x*pt3d_local.x + pt3d_local.y*pt3d_local.y) / (PatchSize*PatchSize/4);

        if ( fabs(pt3d_local.z) < float(PatchSize)*0.2 &&
            dist_2 < 0.8 )
        {
          PtInfo ptInfo;

          // get pixel intensity
          ptInfo.intensity = interpolateKernel<iiMean>( ii, x, y, s );
          ptInfo.grad.x = interpolateKernel<iiDx>( ii, x, y, s );
          ptInfo.grad.y = interpolateKernel<iiDy>( ii, x, y, s );

          // correct for scaling along axis
          ptInfo.grad = affine_mat * ptInfo.grad;

          // the weight will be
          // - lower further away from the center
          // - higher for higher depth (one pixel covers more surface there, and the sampling is more sparse)

          ptInfo.weight = ( 1.0-dist_2 ) * pt3d.z * depth_norm;

          ptInfo.pos = pt3d_local;

          pts.push_back( ptInfo );

          // save gradient info for orientation histogram
          if ( finite(ptInfo.grad.x) && finite(ptInfo.grad.y) )
          {
            PtGradient grad;
            grad.grad_mag = sqrt ( ptInfo.grad.x*ptInfo.grad.x + ptInfo.grad.y*ptInfo.grad.y ) * ptInfo.weight;
            if ( grad.grad_mag > 0 )
            {
              grad.grad_ori = atan2 ( ptInfo.grad.y, ptInfo.grad.x );
              gradients.push_back( grad );
            }
          }
        }
      }
    }
  }

  float kp_ori = dominantOri( gradients );
  kp_ori = 0;

  Point2f kp_ori_vec( cos(kp_ori), sin(kp_ori) );

  cv::Matx22f kp_ori_rot(
      -kp_ori_vec.x,  -kp_ori_vec.y,
      kp_ori_vec.y, - kp_ori_vec.x );

  cv::Mat1f dximg( PatchSize, PatchSize, 0.5f );
  cv::Mat1f dyimg( PatchSize, PatchSize, 0.5f );

  for ( unsigned p=0; p<pts.size(); p++ )
  {
    Point2f pt3d_rot = kp_ori_rot * Point2f( pts[p].pos.x, pts[p].pos.y );
    Point2f grad_rot = kp_ori_rot * Point2f( pts[p].grad.x, pts[p].grad.y );

    pt3d_rot += Point2f(float(PatchSize)*0.5,float(PatchSize)*0.5);

    for ( int y = -0; y<=0; y++ )
    {
      for ( int x = -0; x<=0; x++ )
      {
        float w = 1;//pts[p].weight
        patch( pt3d_rot.y + y, pt3d_rot.x + x ) = pts[p].intensity * w;
        dximg( pt3d_rot.y + y, pt3d_rot.x + x ) = 0.5 + ( grad_rot.x * w );
        dyimg( pt3d_rot.y + y, pt3d_rot.x + x ) = 0.5 + ( grad_rot.y * w );
      }
    }
  }
  imshow2("i",patch);
  imshow2("dx",dximg);
  imshow2("dy",dyimg);
}

}
}

#endif
