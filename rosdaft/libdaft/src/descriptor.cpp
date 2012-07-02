/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include "descriptor.h"

namespace cv
{
namespace daft
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

float dominantOri( std::vector< PtGradient >& gradients, bool show_win )
{
  if ( gradients.size() == 0 )
  {
    return 0.0;
  }

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

  bool win_started = false;

  while ( window_begin < gradients.size() )
  {
      float window_size = ( gradients[window_end].grad_ori + ori_offset ) - gradients[window_begin].grad_ori;
      if ( window_size < M_PI / 3 )
      {
          //found new max.
          if ( win_started )
          {
            if ( grad_mag_sum > grad_max.grad_mag )
            {
                grad_max.grad_mag = grad_mag_sum;
                grad_max.grad_ori = grad_ori_sum;
            }
          }
          window_end++;
          if ( window_end >= gradients.size() )
          {
              window_end = 0;
              ori_offset += 2 * M_PI;
          }
          grad_mag_sum += gradients[window_end].grad_mag;
          grad_ori_sum += gradients[window_end].grad_mag * ( gradients[window_end].grad_ori + ori_offset );

          if ( show_win && win_started )
          {
            hist_values.push_back( std::pair<float,float>( grad_ori_sum/grad_mag_sum , grad_mag_sum ) );
            max_mag = std::max( max_mag, gradients[window_end].grad_mag );
          }
      }
      else
      {
          grad_mag_sum -= gradients[window_begin].grad_mag;
          grad_ori_sum -= gradients[window_begin].grad_mag * gradients[window_begin].grad_ori;
          window_begin++;
          win_started = true;
      }
  }

  float dominant_ori = grad_max.grad_ori / grad_max.grad_mag;

  if ( show_win )
  {
    cv::Mat3b hist_img( 220, 220, Vec3b(255,255,255) );
    Point last_pt;
    Point first_pt;
    for ( std::list< std::pair<float, float> >::iterator it = hist_values.begin(); it != hist_values.end(); ++it )
    {
      float ori = it->first;
      float mag = it->second / grad_max.grad_mag*100;

      Point curr_pt(110+cos(ori)*mag, 110+sin(ori)*mag );
      if ( it == hist_values.begin() )
      {
        first_pt = curr_pt;
      }
      else
      {
        cv::line( hist_img, last_pt, curr_pt, Scalar(0,255,0), 1, 16 );
      }
      last_pt = curr_pt;
    }
    cv::line( hist_img, first_pt, last_pt, Scalar(0,255,0), 1, 16 );
    cv::line( hist_img, Point(110,110), Point(110+cos(dominant_ori)*100, 110+sin(dominant_ori)*100 ), Scalar(0,0,255), 1, 16 );
    imshow( "ori hist", hist_img );

    // debug visualization
    cv::Mat3b grad_img( 440, 440, 0.0f );
    for ( std::vector< PtGradient >::iterator it = gradients.begin(); it != gradients.end(); ++it )
    {
      float ori = it->grad_ori;
      float mag = it->grad_mag/max_mag*200;
      if ( checkBounds( grad_img, 220+sin(ori)*mag, 220+cos(ori)*mag, 1 ) )
      {
        grad_img(220+sin(ori)*mag,220+cos(ori)*mag) = Vec3b(0,255,0);
      }
    }
    cv::line( grad_img, Point(220,220), Point(220+cos(dominant_ori)*200, 220+sin(dominant_ori)*200 ), Scalar(0,0,255), 1, 16 );
    imshow( "ori", grad_img );
  }

  return dominant_ori;
}


Vec3f getNormal( const KeyPoint3D& kp, const cv::Mat1f depth_map, cv::Matx33f& K, float size_mult )
{
  float angle = kp.aff_angle;
  float major = kp.aff_major * 0.25;
  float minor = kp.aff_minor * 0.25;

  int s_round = int( sqrt(major*minor) + 0.5 );
  if ( s_round < 1 ) s_round = 1;

  Matx22f rot_mat;
  rot_mat(0,0) = cos(angle);
  rot_mat(1,0) = sin(angle);
  rot_mat(0,1) = -rot_mat( 1, 0 );
  rot_mat(1,1) = rot_mat( 0, 0 );

  float f_inv = 1.0 / K(0,0);
  float cx = K(0,2);
  float cy = K(1,2);

  std::vector<Vec3f> pts;
  Vec3f mean_pt(0,0,0);

  static const int WIN_SIZE = 3;
  for ( int v = -WIN_SIZE; v<=WIN_SIZE; v++ )
  {
    for ( int u = -WIN_SIZE; u<=WIN_SIZE; u++ )
    {
      Point2f uv( u, v );
      uv = rot_mat.t() * uv;
      uv.x*=major*size_mult;
      uv.y*=minor*size_mult;
      Point2f pixel = rot_mat * uv + kp.pt;

      if ( checkBounds( depth_map, int(pixel.x), int(pixel.y), 1 ) )
      {
        float z = depth_map[int(pixel.y)][int(pixel.x)];
        if ( !isnan(z) )
        {
          Point3f p;
          getPt3d( f_inv, cx, cy, pixel.x, pixel.y, z, p );
          mean_pt += Vec3f(p);
          pts.push_back( Vec3f(p) );
        }
      }
    }
  }

  if ( pts.size() < 3 )
  {
    return Vec3f(kp.normal);
  }

  // center around mean
  mean_pt *= 1.0 / float(pts.size());
  for ( unsigned i = 0; i<pts.size(); i++ )
  {
    pts[i] -= mean_pt;
  }

  Vec3f normal = fitNormal( pts );

  return normal;
}

struct DescEntry
{
  float sum_dx,sum_dy;
  float sum_dx_abs,sum_dy_abs;
  float sum_weights;

  DescEntry() :
    sum_dx(0), sum_dy(0),
    sum_dx_abs(0), sum_dy_abs(0),
    sum_weights(0) {};
  void add( float weight, float dx, float dy )
  {
    dx *= weight;
    dy *= weight;
    sum_dx += dx;
    sum_dy += dy;
    sum_dx_abs += std::abs(dx);
    sum_dy_abs += std::abs(dy);
    sum_weights += weight;
  }
  bool normalize()
  {
    if ( sum_weights == 0 )
    {
      return false;
    }

    sum_dx /= sum_weights;
    sum_dy /= sum_weights;
    sum_dx_abs /= sum_weights;
    sum_dy_abs /= sum_weights;

    return true;
  }
};

bool computeDesc( const vector<PtInfo>& ptInfos, cv::Mat1f& desc, bool show_win )
{
  DescEntry entries[4][4];
  for( unsigned i=0; i<ptInfos.size(); i++ )
  {
    // maps [-1,1] to [-1,4]
    float u = ptInfos[i].pos.x * 2.5 + 1.5;
    float v = ptInfos[i].pos.y * 2.5 + 1.5;
    int ui = std::floor(u);
    int vi = std::floor(v);
    float tu = u - float(ui);
    float tv = v - float(vi);
    const float& dx = ptInfos[i].grad.x;
    const float& dy = ptInfos[i].grad.y;

    /*
    std::cout << "x " << ptInfos[i].pos.x << " y " << ptInfos[i].pos.y << std::endl;
    std::cout << "u " << u << " v " << v << std::endl;
    std::cout << "ui " << ui << " vi " << vi << std::endl;*/

    if ( ui >= 0 && ui < 4 )
    {
      if ( vi >= 0 && vi < 4 )
      {
        entries[vi][ui].add( (1.0f-tu)*(1.0-tv), dx, dy );
      }
      if ( vi+1 >= 0 && vi+1 < 4 )
      {
        entries[vi+1][ui].add( (1.0f-tu)*tv, dx, dy );
      }
    }
    if ( ui+1 >= 0 && ui+1 < 4 )
    {
      if ( vi >= 0 && vi < 4 )
      {
        entries[vi][ui+1].add( tu*(1.0-tv), dx, dy );
      }
      if ( vi+1 >= 0 && vi+1 < 4 )
      {
        entries[vi+1][ui+1].add( tu*tv, dx, dy );
      }
    }
  }

  assert( desc.cols == 64 );
  int idx=0;

  for ( unsigned j=0; j<4; j++ )
  {
    for ( unsigned i=0; i<4; i++ )
    {
      // disallow 4 central bins to be empty
      if ( !entries[j][i].normalize() )// && i>=1 && i<=2 && j>=1 && j<=2 )
      {
        return false;
      }

      desc(idx++) = entries[j][i].sum_dx;
      desc(idx++) = entries[j][i].sum_dy;
      desc(idx++) = entries[j][i].sum_dx_abs;
      desc(idx++) = entries[j][i].sum_dy_abs;
    }
  }

  //normalize descriptor vector
  float l2=0;
  for ( int i=0; i<desc.cols; i++ )
  {
    l2+=desc(i)*desc(i);
  }
  float l=sqrt(l2);
  for ( int i=0; i<desc.cols; i++ )
  {
    desc(i) /= l;
  }

  if ( show_win )
  {
    cv::Mat1f dx_img(4,4);
    cv::Mat1f dy_img(4,4);
    cv::Mat1f dx_abs_img(4,4);
    cv::Mat1f dy_abs_img(4,4);
    cv::Mat1f weight_img(4,4);

    float max_w = 0.0;

    for ( unsigned j=0; j<4; j++ )
    {
      for ( unsigned i=0; i<4; i++ )
      {
        max_w = std::max( max_w, entries[j][i].sum_weights );
        weight_img[j][i] = entries[j][i].sum_weights;
        dx_img[j][i] = entries[j][i].sum_dx;
        dy_img[j][i] = entries[j][i].sum_dy;
        dx_abs_img[j][i] = entries[j][i].sum_dx_abs;
        dy_abs_img[j][i] = entries[j][i].sum_dy_abs;
      }
    }

    imshow2( "desc_weights", weight_img * (1.0/max_w) );
    imshowNorm( "desc_dx", dx_img );
    imshowNorm( "desc_dy", dy_img );
    imshowNorm( "desc_dx_abs", dx_abs_img );
    imshowNorm( "desc_dy_abs", dy_abs_img );
  }

  return true;
}


inline float Gaussian2(float sigma, float d2) {
  return 0.39894228f / sigma * std::exp(-0.5f * d2 / (sigma * sigma));
}

inline float Gaussian2(float sigma, float x, float y) {
  return 1.0/3.00694*0.73469*Gaussian2(sigma, x*x+y*y);
}
inline float Gaussian2(float sigma, float x, float y, float z) {
  return 1.0/3.00694*0.73469*Gaussian2(sigma, x*x+y*y+z*z);
}

template< int Sigma3 >
inline void getGradPatch( int patch_size, float thickness, Mat1f& smoothed_img, const KeyPoint3D& kp,
    const cv::Mat1f depth_map, cv::Matx33f& K, vector<PtInfo>& ptInfos, float& coverage,
    float step_size, cv::Matx33f kp_ori_rot, float max_dist_2,
    bool show_win=false )
{
  float sum_weights = 0;

  static const float Sigma = float(Sigma3) / 3.0;
  static const float nan = std::numeric_limits<float>::quiet_NaN();
  Mat1f patch( patch_size+1, patch_size+1, nan );

  //// debug visualization

  Mat1f patch_dx( patch_size, patch_size, nan );
  Mat1f patch_dy( patch_size, patch_size, nan );
  Mat1f patch_weights( patch_size, patch_size, 0.0f );

  static const int PATCH_MUL = 32;
  Mat3b points3d_img( patch_size*PATCH_MUL, patch_size*PATCH_MUL, Vec3b( 255,255,255 ) );

  Mat display_img;

  if ( show_win )
  {
    // hack: only do this for the descriptor window
    if ( Sigma3 == 1 )
    {
      for ( int v=0; v<points3d_img.rows; v++ )
      {
        for ( int u=0; u<points3d_img.rows; u++ )
        {
          if ( ( (v+2+patch_size*PATCH_MUL/10) % (patch_size*PATCH_MUL/5) < 3 ) ||
               ( (u+2+patch_size*PATCH_MUL/10) % (patch_size*PATCH_MUL/5) < 3 ) )
          {
            points3d_img[v][u] = Vec3b( 200,200,200 );
          }
        }
      }
    }

    Mat tmp;
    smoothed_img.convertTo( tmp, CV_8U, 255, 0 );
    cvtColor( tmp, display_img, CV_GRAY2RGB );
  }

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

  // transforms from local 3d coords to 3d
  cv::Matx33f local_to_cam( -v1.x, -v2.x, normal.x, -v1.y, -v2.y, normal.y, -v1.z, -v2.z, normal.z );
  local_to_cam = local_to_cam * kp_ori_rot.t();

  // transforms from u/v texture coords [-PatchSize/2 ... PatchSize/2] to 3d
  cv::Matx33f uvw_to_cam = local_to_cam * kp.world_size * 0.5 * step_size;

  // transforms from 3d to u/v tex coords
  float s1 = 2.0 / (step_size*kp.world_size);
  cv::Matx33f cam_to_uvw = cv::Matx33f(s1,0,0, 0,s1,0, 0,0,s1) * local_to_cam.t();

  // sample intensity values using planar assumption
  for ( int v = 0; v<patch_size+1; v++ )
  {
    for ( int u = 0; u<patch_size+1; u++ )
    {
      // compute u-v coordinates
      Point2f uv( float(u)-float(patch_size/2), float(v-float(patch_size/2)) );

      /*
      const float dist2 = (uv.x*uv.x + uv.y*uv.y);
      if ( dist2 > float((PatchSize+1)*(PatchSize+1)) * 0.25f )
      {
        continue;
      }
      */

      Point2f pixel;
      Point3f pt_cam = (uvw_to_cam * Point3f(uv.x,uv.y,0)) + kp.pt3d;
      getPt2d( pt_cam, K(0,0), cx, cy, pixel );

      if ( checkBounds( smoothed_img, pixel.x, pixel.y, 1 ) )
      {
        patch[v][u] = interpBilinear(smoothed_img,pixel.x,pixel.y);

        if ( show_win && !isnan( patch[v][u] ) )
        {
          float s = 0.5 * kp.world_size * K(0,0) / depth_map(int(pixel.y),int(pixel.x));
          Size2f bsize( s, kp.aff_minor/kp.aff_major*s );
          cv::RotatedRect box(pixel, bsize, kp.aff_angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,0,255), 1, CV_AA );
        }
      }
    }
  }

  //cv::Matx22f kp_ori_rot_2d( kp_ori_rot(0,0), kp_ori_rot(0,1), kp_ori_rot(1,0), kp_ori_rot(1,1) );

  // coordinate in u/v of window center
  const float center_uv = (float(patch_size)-1.0f) * 0.5;

  if ( show_win )
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

  ptInfos.clear();
  ptInfos.reserve(patch_size*patch_size);

  const float thickness2 = thickness*thickness;

  // compute gradients
  for ( int v = 0; v<patch_size; v++ )
  {
    for ( int u = 0; u<patch_size; u++ )
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
      Point2f uv_reproj( pt3d_uvw.x + float(patch_size/2), pt3d_uvw.y + float(patch_size/2) );

      // normalized patch coords [-1,1]
      Point3f pt3d_uvw1 = pt3d_uvw * (2.0f/float(patch_size));
      float dist_2 = pt3d_uvw1.x*pt3d_uvw1.x + pt3d_uvw1.y*pt3d_uvw1.y + (pt3d_uvw1.z*pt3d_uvw1.z)/(thickness2);

      if ( isnan(dist_2) || dist_2 > max_dist_2 )
      {
        continue;
      }

      const float weight = Gaussian2(Sigma, dist_2);


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

        if ( show_win )
        {
          patch_weights[v][u] = weight;
          patch_dx[v][u] = dx;
          patch_dy[v][u] = dy;

          if (!isnan(val))
          {
            float circle_size = 1.0 - ( pt3d_uvw1.z/thickness );
            Size2f bsize( float(PATCH_MUL)*circle_size,float(PATCH_MUL)*circle_size );
            //Size2f bsize( float(PATCH_MUL)*1.0,float(PATCH_MUL)*1.0 );
            //Size2f bsize( float(PATCH_MUL),float(PATCH_MUL) );
            cv::RotatedRect box(uv_reproj*PATCH_MUL, bsize, kp.aff_angle/M_PI*180.0 );
            ellipse( points3d_img, box, cv::Scalar(val*255,val*255,val*255),-1, CV_AA );
            //ellipse( points3d_img, box, cv::Scalar(128,0,0),1, CV_AA );
          }

          Size2f bsize( 3,3 );
          cv::RotatedRect box(pixel, bsize, kp.aff_angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,255,0), 1, CV_AA );
        }
      }
    }
  }

  if ( show_win )
  {
    /*
    Size2f bsize( kp.affine_major, kp.affine_minor );
    cv::RotatedRect box( kp.pt, bsize, kp.affine_angle/M_PI*180.0 );
    ellipse( display_img, box, cv::Scalar(0,0,255), 1, CV_AA );
    */

    std::ostringstream s;
    s << patch_size << "-" << Sigma3 << " ";
    std::string prefix = s.str();

    imshow( prefix+"sampling points", display_img );
    imshow2( prefix+"affine patch", patch );
    imshow2( prefix+"affine patch dx", patch_dx*5 + 0.5 );
    imshow2( prefix+"affine patch dy", patch_dy*5 + 0.5 );
    imshowNorm( prefix+"weights", patch_weights );
    imshow( prefix+"3d points", points3d_img );
  }

  coverage = sum_weights / float(patch_size*patch_size) * 13.5;
  //static float max_coverage = 0.0;
  //max_coverage = std::max(coverage,max_coverage);
  //std::cout << "max_coverage: " << max_coverage << std::endl;
}



bool SurfDescriptor::getDesc(
    int patch_size,
    float thickness,
    Mat1f& smoothed_img,
    Mat1f& smoothed_img2,
    KeyPoint3D& kp,
    Mat1f& desc,
    const cv::Mat1f depth_map,
    cv::Matx33f& K,
    bool show_win )
{
  // get gradients from larger scale
  std::vector<PtInfo> pt_infos_ori;
  float coverage;
  getGradPatch<1>( patch_size, thickness*2.0, smoothed_img2, kp, depth_map, K, pt_infos_ori, coverage, 0.5, cv::Matx33f::eye(), 1.0, show_win );

  if ( coverage < 0.5 * 0.86 )
  {
    //std::cout << "orientation coverage: " << coverage << std::endl;
    return false;
  }

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
  float kp_ori = dominantOri( gradients, show_win );

  Point2f kp_ori_vec( cos(kp_ori), sin(kp_ori) );

  cv::Matx33f kp_ori_rot(
      -kp_ori_vec.x,  -kp_ori_vec.y, 0,
      kp_ori_vec.y, - kp_ori_vec.x, 0,
      0,0,1 );

  std::vector<PtInfo> pt_infos_desc;
  getGradPatch<2>( patch_size, thickness, smoothed_img, kp, depth_map, K, pt_infos_desc, coverage, 1.0, kp_ori_rot, 2.0, show_win );

  if ( coverage < 0.5 )
  {
    //std::cout << "descriptor coverage: " << coverage << std::endl;
    return false;
  }

  return computeDesc( pt_infos_desc, desc, show_win );
}

}}

