/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_DESCRIPTOR_H__
#define __DAFT2_DESCRIPTOR_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <features3d/keypoint3d.h>

#include "filter_kernels.h"
#include "stuff.h"

#include <math.h>
#include <list>


#define DESC_DEBUG_IMG

namespace cv
{
namespace daft2
{

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

float dominantOri( std::vector< PtGradient >& gradients );

Vec3f getNormal( const KeyPoint3D& kp, const cv::Mat1f depth_map, cv::Matx33f& K );

void computeDesc( const vector<PtInfo>& ptInfos, std::vector<float>& desc );

template< int PatchSize, int Sigma3 >
inline void getGradPatch( Mat1f& smoothed_img, const KeyPoint3D& kp,
    const cv::Mat1f depth_map, cv::Matx33f& K, vector<PtInfo>& ptInfos, float& coverage,
    float step_size=1, cv::Matx33f kp_ori_rot = cv::Matx33f::eye() )
{
  float sum_weights = 0;

  static const float Sigma = float(Sigma3) / 3.0;
  static const float nan = std::numeric_limits<float>::quiet_NaN();
  Mat1f patch( PatchSize+1, PatchSize+1, nan );

#ifdef DESC_DEBUG_IMG
  Mat1f patch_dx( PatchSize, PatchSize, nan );
  Mat1f patch_dy( PatchSize, PatchSize, nan );
  Mat1f patch_weights( PatchSize, PatchSize, 1.0f );

  static const int PATCH_MUL = 32;
  Mat3b points3d_img( PatchSize*PATCH_MUL, PatchSize*PATCH_MUL, Vec3b( 255,255,255 ) );

  // hack: only do this for the descriptor window
  if ( Sigma3 == 1 )
  {
    for ( int v=0; v<points3d_img.rows; v++ )
    {
      for ( int u=0; u<points3d_img.rows; u++ )
      {
        if ( ( (v+2+PatchSize*PATCH_MUL/10) % (PatchSize*PATCH_MUL/5) < 3 ) ||
             ( (u+2+PatchSize*PATCH_MUL/10) % (PatchSize*PATCH_MUL/5) < 3 ) )
        {
          points3d_img[v][u] = Vec3b( 200,200,200 );
        }
      }
    }
  }

  Mat tmp, display_img;
  smoothed_img.convertTo( tmp, CV_8U, 255, 0 );
  cvtColor( tmp, display_img, CV_GRAY2RGB );
#endif

  //const Point2f& pt
  float angle = kp.affine_angle;
  float major = kp.affine_major*0.5;
  float minor = kp.affine_minor*0.5;

  // camera params
  float f_inv = 1.0 / K(0,0);
  float cx = K(0,2);
  float cy = K(1,2);

  Point3f affine_major_axis( 0,-1,0 );//cos(angle), sin(angle), 0 );

  Vec3f n = kp.normal;

  Point3f normal(n[0],n[1],n[2]);
  Point3f v1 = normal.cross( affine_major_axis );
  v1 = v1  * fastInverseLen( v1 );
  Point3f v2 = v1.cross( normal );

  // transforms from local 3d coords [-0.5...0.5] to 3d
  cv::Matx33f local_to_cam( -v1.x, -v2.x, normal.x, -v1.y, -v2.y, normal.y, -v1.z, -v2.z, normal.z );
  local_to_cam = local_to_cam * kp_ori_rot.t();

  // transforms from u/v texture coords [-PatchSize/2 ... PatchSize/2] to 3d
  cv::Matx33f uvw_to_cam = local_to_cam * kp.world_size * 0.25 * step_size;

  // transforms from 3d to u/v tex coords
  cv::Matx33f cam_to_uvw = cv::Matx33f(4.0 / (step_size*kp.world_size),0,0, 0,4.0 / kp.world_size,0, 0,0,1) * local_to_cam.t();

  // sample intensity values using planar assumption
  for ( int v = 0; v<PatchSize+1; v++ )
  {
    for ( int u = 0; u<PatchSize+1; u++ )
    {
      // compute u-v coordinates
      Point2f uv( float(u)-float(PatchSize/2), float(v-float(PatchSize/2)) );

      const float dist2 = (uv.x*uv.x + uv.y*uv.y);
      if ( dist2 > float((PatchSize+1)*(PatchSize+1)) * 0.25f )
      {
        continue;
      }

      Point2f pixel;
      Point3f pt_cam = (uvw_to_cam * Point3f(uv.x,uv.y,0)) + kp.pt3d;
      getPt2d( pt_cam, K(0,0), cx, cy, pixel );

      if ( checkBounds( smoothed_img, pixel.x, pixel.y, 1 ) )
      {
        patch[v][u] = interpBilinear(smoothed_img,pixel.x,pixel.y);

#ifdef DESC_DEBUG_IMG
        if ( !isnan( patch[v][u] ) )
        {
          float s = 0.5 * kp.world_size * K(0,0) / depth_map(int(pixel.y),int(pixel.x));
          Size2f bsize( s, minor/major*s );
          cv::RotatedRect box(pixel, bsize, angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,0,255), 1, CV_AA );
        }
#endif
      }
    }
  }

  //cv::Matx22f kp_ori_rot_2d( kp_ori_rot(0,0), kp_ori_rot(0,1), kp_ori_rot(1,0), kp_ori_rot(1,1) );

  // coordinate in u/v of window center
  const float center_uv = (float(PatchSize)-1.0f) * 0.5;

#ifdef DESC_DEBUG_IMG
  {
    Point3f pn3d = kp.pt3d + (local_to_cam * Point3f(0,0,1)*kp.world_size);
    Point3f pv13d = kp.pt3d + (local_to_cam * Point3f(1,0,0)*kp.world_size);
    Point3f pv23d = kp.pt3d + (local_to_cam * Point3f(0,1,0)*kp.world_size);

    Point2f p,pn,pv1,pv2;

    getPt2d( pn3d, (float)K(0,0), cx, cy, pn );
    getPt2d( pv13d, (float)K(0,0), cx, cy, pv1 );
    getPt2d( pv23d, (float)K(0,0), cx, cy, pv2 );
    getPt2d( kp.pt3d, (float)K(0,0), cx, cy, p );

    cv::line( display_img, p, pv1, cv::Scalar( 0,0,255 ), 2, 16 );
    cv::line( display_img, p, pv2, cv::Scalar( 0,255,0 ), 2, 16 );
    cv::line( display_img, p, pn, cv::Scalar( 255,0,0 ), 2, 16 );
  }
#endif

  ptInfos.clear();
  ptInfos.reserve(PatchSize*PatchSize);

  // compute gradients
  for ( int v = 0; v<PatchSize; v++ )
  {
    for ( int u = 0; u<PatchSize; u++ )
    {
      // 0-centered u/v coords
      Point2f uv( float(u)-center_uv, float(v)-center_uv );

      Point2f pixel;
      Point3f pt_cam = (uvw_to_cam * Point3f(uv.x,uv.y,0)) + kp.pt3d;
      getPt2d( pt_cam, K(0,0), cx, cy, pixel );

      if ( !checkBounds( smoothed_img, pixel.x, pixel.y, 1 ) )
      {
        continue;
      }

      // get the 3d coordinates of the points
      Point3f pt3d;
      int pixel_x_int = pixel.x + 0.5;
      int pixel_y_int = pixel.y + 0.5;
      getPt3d( f_inv, cx, cy, pixel.x, pixel.y, depth_map[pixel_y_int][pixel_x_int], pt3d );

      // local 3d tex coords of point [-PatchSize/2,PatchSize/2]
      Point3f pt3d_uvw = cam_to_uvw * (pt3d - kp.pt3d);

      // uv indices of reprojected 3d point [0,PatchSize]
      Point2f uv_reproj( pt3d_uvw.x + float(PatchSize/2), pt3d_uvw.y + float(PatchSize/2) );

      // normalized patch coords [-1,1]
      Point3f pt3d_uvw1 = pt3d_uvw * (2.0f/float(PatchSize));
      float dist_2 = pt3d_uvw1.x*pt3d_uvw1.x + pt3d_uvw1.y*pt3d_uvw1.y + 3*pt3d_uvw1.z*pt3d_uvw1.z;

      const float weight = 1.0 - dist_2;
      if ( isnan(weight) || weight <= 0.0 )
      {
        continue;
      }

      sum_weights += weight;

      float dx = 0.5 * weight * ( patch[v][u+1] + patch[v+1][u+1] - patch[v][u] - patch[v+1][u] );
      float dy = 0.5 * weight * ( patch[v+1][u] + patch[v+1][u+1] - patch[v][u] - patch[v][u+1] );

      if ( !isnan(dx) && !isnan(dy) )
      {
        PtInfo ptInfo;
        ptInfo.grad = Point2f(dx,dy);
        ptInfo.weight = weight;
        ptInfo.pos = pt3d_uvw1;

        float val = smoothed_img[int(pixel.y)][int(pixel.x)];
        ptInfo.intensity = val;

        ptInfos.push_back( ptInfo );

#ifdef DESC_DEBUG_IMG
        Size2f bsize( float(PATCH_MUL)*weight,float(PATCH_MUL)*weight );
        cv::RotatedRect box(uv_reproj*PATCH_MUL, bsize, angle/M_PI*180.0 );
        ellipse( points3d_img, box, cv::Scalar(val*255,val*255,val*255),-1, CV_AA );
        //ellipse( points3d_img, box, cv::Scalar(128,0,0),1, CV_AA );

        patch_weights[v][u] = weight;
        patch_dx[v][u] = dx;
        patch_dy[v][u] = dy;

        if ( !isnan( patch_dx[v][u] ) && !isnan( patch_dy[v][u] ) )
        {
          Size2f bsize( 3,3 );
          cv::RotatedRect box(pixel, bsize, angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,255,0), 1, CV_AA );
        }
#endif
      }
    }
  }

#ifdef DESC_DEBUG_IMG
  /*
  Size2f bsize( kp.affine_major, kp.affine_minor );
  cv::RotatedRect box( kp.pt, bsize, kp.affine_angle/M_PI*180.0 );
  ellipse( display_img, box, cv::Scalar(0,0,255), 1, CV_AA );
  */

  std::ostringstream s;
  s << PatchSize << "-" << Sigma3 << " ";
  std::string prefix = s.str();

  imshow( prefix+"sampling points", display_img );
  imshow2( prefix+"affine patch", patch );
  imshow2( prefix+"affine patch dx", patch_dx*5 + 0.5 );
  imshow2( prefix+"affine patch dy", patch_dy*5 + 0.5 );
  imshow2( prefix+"weights", patch_weights );
  imshow( prefix+"3d points", points3d_img );
#endif

  coverage = sum_weights / float(PatchSize*PatchSize) / 0.4;
  std::cout << "coverage: " << coverage << std::endl;
}


template< int OriPatchSize, int DescPatchSize >
inline bool getDesc( Mat1f& smoothed_img, Mat1f& smoothed_img2, KeyPoint3D& kp, const cv::Mat1f depth_map, cv::Matx33f& K )
{
  static const float nan = std::numeric_limits<float>::quiet_NaN();

  // compute exact normal unsing pca
  kp.normal = getNormal(kp, depth_map, K );

  // get gradients from larger scale
  std::vector<PtInfo> pt_infos_ori;
  float coverage;
  getGradPatch<DescPatchSize,2>( smoothed_img2, kp, depth_map, K, pt_infos_ori, coverage, 0.5 );

  // construct gradient vector
  std::vector< PtGradient > gradients;
  gradients.reserve(pt_infos_ori.size());

  for ( unsigned i=0; i<pt_infos_ori.size(); i++ )
  {
    PtInfo& pt_info = pt_infos_ori[i];
    PtGradient grad;
    grad.grad_mag = sqrt ( pt_info.grad.x*pt_info.grad.x + pt_info.grad.y*pt_info.grad.y ) * pt_info.weight;
    if ( grad.grad_mag > 0 )
    {
      grad.grad_ori = atan2 ( pt_info.grad.y, pt_info.grad.x );
      gradients.push_back( grad );
    }
  }

  // get dominant orientation
  float kp_ori = dominantOri( gradients );

  Point2f kp_ori_vec( cos(kp_ori), sin(kp_ori) );

  cv::Matx33f kp_ori_rot(
      -kp_ori_vec.x,  -kp_ori_vec.y, 0,
      kp_ori_vec.y, - kp_ori_vec.x, 0,
      0,0,1 );

  std::vector<PtInfo> pt_infos_desc;
  getGradPatch<DescPatchSize,1>( smoothed_img, kp, depth_map, K, pt_infos_desc, coverage, 1.0, kp_ori_rot );

  computeDesc( pt_infos_desc, kp.desc );
  return true;
}


}}

#endif
