/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include "descriptor.h"

namespace cv
{
namespace daft2
{

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
  float angle = kp.affine_angle;
  float major = kp.affine_major * 0.25;
  float minor = kp.affine_minor * 0.25;

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

bool computeDesc( const vector<PtInfo>& ptInfos, std::vector<float>& desc, bool show_win )
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

  desc.clear();
  desc.reserve(64);
  for ( unsigned j=0; j<4; j++ )
  {
    for ( unsigned i=0; i<4; i++ )
    {
      if ( !entries[j][i].normalize() )
      {
        return false;
      }

      desc.push_back(entries[j][i].sum_dx);
      desc.push_back(entries[j][i].sum_dy);
      desc.push_back(entries[j][i].sum_dx_abs);
      desc.push_back(entries[j][i].sum_dy_abs);
    }
  }

  //normalize descriptor vector
  float l2=0;
  for ( unsigned i=0; i<desc.size(); i++ )
  {
    l2+=desc[i]*desc[i];
  }
  float l=sqrt(l2);
  for ( unsigned i=0; i<desc.size(); i++ )
  {
    desc[i] /= l;
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


}}

