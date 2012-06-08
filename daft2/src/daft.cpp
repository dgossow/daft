/*
 * DAFT
 * Copyright (C) 2011 David Gossow
 */

#include "daft.h"
#include "filter_kernels.h"
#include "depth_filter.h"
#include "feature_detection.h"
#include "descriptor.h"
#include "preprocessing.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/timer.hpp>

#include <cmath>
#include <set>
#include <map>

namespace cv {
namespace daft2 {

boost::timer t;
std::string dbg_msg;

//DEBUG FLAGS (To be removed) ----
//
//#define SHOW_DEBUG_WIN
//#define FIND_MAXKP
//#define SHOW_MASK
//#define SHOW_PYR
//#define SHOW_RESPONSE

#define DBG_OUT( SEQ ) std::cout << SEQ << std::endl
#define TIMER_STOP if(dbg_msg.length()!=0) { DBG_OUT( "++++++ " << dbg_msg << ": " << t.elapsed()*1000.0 << "ms ++++++" ); }
#define TIMER_START( MSG ) TIMER_STOP dbg_msg=MSG; t.restart();


DAFT::DAFT(const DetectorParams & detector_params, const DescriptorParams & descriptor_params ) :
    det_params_(detector_params), desc_params_(descriptor_params)
{
}

DAFT::~DAFT()
{
}

void DAFT::operator()( const cv::Mat &image, const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp, cv::Mat1f& desc )
{
  cv::Mat1b mask( (int)image.rows, (int)image.cols, (cv::Mat1b::value_type)255 );
  computeImpl(image,mask,depth_map_orig,K,kp,desc,true);
}

void DAFT::operator()( const cv::Mat &image, const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp )
{
  cv::Mat1b mask( (int)image.rows, (int)image.cols, (cv::Mat1b::value_type)255 );
  cv::Mat1f desc;
  computeImpl(image,mask,depth_map_orig,K,kp,desc,false);
}

void DAFT::operator()( const cv::Mat &image, const cv::Mat1b &mask,
    const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp, cv::Mat1f& desc )
{
  computeImpl(image,mask,depth_map_orig,K,kp,desc,true);
}

void DAFT::operator()( const cv::Mat &image, const cv::Mat1b &mask,
    const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp )
{
  cv::Mat1f desc;
  computeImpl(image,mask,depth_map_orig,K,kp,desc,false);
}

void DAFT::computeImpl(
    const cv::Mat &image,
    const cv::Mat1b &mask_orig,
    const cv::Mat &depth_map_orig,
    cv::Matx33f K,
    std::vector<KeyPoint3D> & kp,
    cv::Mat1f& desc,
    bool computeDescriptors )
{
  boost::timer t2;

  kp.clear();
  desc = cv::Mat1f();

  TIMER_START( "Data Preparation" );

  if (image.size != depth_map_orig.size) {
    return;
  }

  // Convert input to correct format
  Mat gray_image;
  Mat1d ii;
  Mat1f depth_map;
  cv::Mat1b mask = mask_orig.clone();

  prepareData( image, depth_map_orig, gray_image, ii, depth_map, mask );

  // put an upper limit on the max. pixel scale
  float max_px_scale = 4.0 * std::min(image.rows, image.cols) / desc_params_.patch_size_ / pow(2.0,desc_params_.octave_offset_);

  if ( det_params_.max_px_scale_ != det_params_.AUTO )
  {
    max_px_scale = std::min( max_px_scale, det_params_.max_px_scale_ );
  }

  const float f = K(0, 0);

  // compute scale map (inverse depth map)
  Mat1f scale_map(gray_image.rows, gray_image.cols);
  computeScaleMap( depth_map, mask, f, scale_map );

  // make list of all octaves that we need to compute
  std::set<int> pyr_octaves;
  std::set<int> det_octaves;
  getOctaves( scale_map, max_px_scale, pyr_octaves, det_octaves );

  kp.clear();
  kp.reserve(50000);

  TIMER_START( "Affine Parameters" );

  //std::map< int, Mat1f > smoothed_imgs;
  std::map< int, Mat1f> smoothed_depth_maps;
  std::map< int, Mat3f > affine_maps; // entries are (major_len, minor_len, major_x, major_y)

  computeAffineMaps( pyr_octaves, depth_map, scale_map, f, smoothed_depth_maps, affine_maps );

  // compute depth-normalized image pyramid
  TIMER_START( "Gaussian Pyramid" );

  for (std::set<int>::iterator it = pyr_octaves.begin(); it != pyr_octaves.end(); it++ )
  {
    int octave = *it;
    double scale = det_params_.base_scale_ * std::pow( 2.0, float(octave) );
    Mat1f& smoothed_img = smoothed_imgs[octave];
    Mat3f& affine_map = affine_maps[octave];

    // compute filter response for all pixels
    switch (det_params_.det_type_) {
    case DetectorParams::DET_FELINE:
      if (det_params_.affine_) {
        convolveAffine<feline>(ii, scale_map, affine_map, scale, 1, smoothed_img );
      } else {
        convolve<boxMean>(ii, scale_map, scale, 1, smoothed_img);
      }
      break;
    default:
      DBG_OUT( "error: invalid detector type: " << det_params_.det_type_ );
      return;
    }

#ifdef SHOW_PYR
    {
      s.str("");
      s << "Smoothed Image - Detector type=" << det_params_.det_type_ << " max=" << det_params_.max_search_algo_ << " scale = " << scale << " affine = " << det_params_.affine_;
      imshow( s.str(), smoothed_img );
    }
#endif
  }

  TIMER_START( "Computing response & finding max" );

  // compute abs. difference of gaussians and detect maxima
  for (std::set<int>::iterator it = det_octaves.begin(); it != det_octaves.end(); it++ )
  {
    int octave = *it;
    double scale = det_params_.base_scale_ * std::pow( det_params_.scale_step_, float(octave) );

    response_maps[octave] = smoothed_imgs[octave+1] - smoothed_imgs[octave];

    Mat3f& affine_map = affine_maps[octave];
    Mat1f &response_map = response_maps[octave];

    std::vector< KeyPoint3D > kp_init;

    // find maxima in response
    switch (det_params_.max_search_algo_) {
    case DetectorParams::MAX_WINDOW:
      if ( det_params_.affine_ ) {
        findExtremaAffine(response_map, scale_map, affine_map,
            scale, det_params_.min_px_scale_, max_px_scale,
            det_params_.min_dist_,
            det_params_.det_threshold_, kp_init);
      }
      else {
        findExtrema(response_map, scale_map, scale,
            det_params_.min_px_scale_, max_px_scale,
            det_params_.min_dist_,
            det_params_.det_threshold_, kp_init);
      }
      break;
    case DetectorParams::MAX_FAST:
      findMaximaMipMap(response_map, scale_map, scale,
          det_params_.min_px_scale_, max_px_scale,
          det_params_.det_threshold_, kp_init);
      break;
    default:
      DBG_OUT( "error: invalid max search type: " << det_params_.max_search_algo_ );
      return;
    }

    DBG_OUT( "octave " << octave << " scale " << scale << ": " << kp_init.size() << " keypoints found." );

    // store octave
    for ( unsigned k=0; k<kp_init.size(); k++ )
    {
      kp_init[k].octave = octave;
    }

    // principal curvature filter
    int old_kp_size = kp.size();
    princCurvFilter( response_map, scale_map, affine_map, det_params_.max_princ_curv_ratio_, kp_init, kp );
    DBG_OUT( "octave " << octave << " scale " << scale << ": " << kp.size()-old_kp_size << " keypoints left after principal curvature filter." );

#ifdef SHOW_RESPONSE
    {
      static int i=0;
      cv::Mat display_image;
      response_map.convertTo( display_image, CV_8UC1, 512, 128 );

      std::vector<KeyPoint3D> kp2;
      princCurvFilter( response_map, scale_map, affine_map, det_params_.max_princ_curv_ratio_, kp_init, kp2 );

      cv::drawKeypoints3D( display_image, kp_init, display_image, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      cv::drawKeypoints3D( display_image, kp2, display_image, cv::Scalar(255,0,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      s.str("");
      s << "Response - Detector type=" << det_params_.det_type_ << " max=" << det_params_.max_search_algo_ << " scale = " << scale << " affine = " << det_params_.affine_;
      imshow( s.str(), cv::Mat( display_image ) );
      i++;
    }
#endif
  }

  DBG_OUT( kp.size() << " keypoints found." );

  TIMER_START("Descriptor computation");

  // if we have a limitation on the number of keypoints, sort them first
  if ( det_params_.max_num_kp_ != std::numeric_limits<unsigned>::max() )
  {
    std::sort( kp.begin( ), kp.end( ), compareResponse<KeyPoint3D> );
  }

  // assign 3d points, normals and local affine params
  float f_inv = 1.0 / f;
  float cx = K(0, 2);
  float cy = K(1, 2);

  std::vector<KeyPoint3D> kp2;

  kp2.reserve(kp.size());

  SurfDescriptor surf_desc;
  cv::Mat1f desc_tmp( kp.size(), surf_desc.getDescLen() );

  // correct keypoint size according to given octave offset
  float scale_fac = pow( 2.0, float(desc_params_.octave_offset_) );

  float min_size = det_params_.min_px_scale_*det_params_.min_px_scale_;

  for (unsigned k = 0; k < kp.size() && kp2.size() < det_params_.max_num_kp_; k++)
  {
  	kp[k].size *= scale_fac;
  	kp[k].world_size *= scale_fac;
  	kp[k].octave += desc_params_.octave_offset_;

    int kp_x = kp[k].pt.x;
    int kp_y = kp[k].pt.y;
    getPt3d(f_inv, cx, cy, kp_x, kp_y, depth_map[kp_y][kp_x], kp[k].pt3d);

    Vec3f affine_params = affine_maps[kp[k].octave](kp_y,kp_x);

    if ( mask( kp[k].pt.y, kp[k].pt.x ) != 0 &&
         kp[k].size > min_size )
    {
      kp[k].aff_minor = affine_params[0] * kp[k].aff_major;
      kp[k].aff_angle = atan2( affine_params[2], affine_params[1] );

      // compute exact normal using pca
      kp[k].normal = getNormal(kp[k], depth_map, K, 2.0 / scale_fac );

      if ( !computeDescriptors )
      {
        kp2.push_back(kp[k]);
      }
      else
      {
        Mat1f& smoothed_img = smoothed_imgs[kp[k].octave];
        Mat1f& smoothed_img2 = smoothed_imgs[kp[k].octave+1];

        // reference current row from descriptor matrix
        Mat1f curr_desc(desc_tmp, cv::Rect( 0, kp2.size(), desc_tmp.cols, 1 ) );

        if ( surf_desc.getDesc( desc_params_.patch_size_, desc_params_.z_thickness_, smoothed_img, smoothed_img2, kp[k], curr_desc, depth_map, K ) )
        {
          kp2.push_back(kp[k]);
        }
      }
    }
  }
  kp.swap(kp2);

  // copy descriptors to matrix
  desc = desc_tmp( cv::Rect( 0, 0, surf_desc.getDescLen(), kp.size() ) ).clone();

  DBG_OUT( kp.size() << " keypoints left after descriptor computation." );

  DBG_OUT( "++++++ Total time elapsed: " << t2.elapsed()*1000.0 << "ms ++++++" );
}

void DAFT::computeAffineMaps(
    std::set<int>& octaves,
    cv::Mat1f& depth_map,
    cv::Mat1f& scale_map,
    float f,
    std::map< int, Mat1f>& smoothed_depth_maps,
    std::map< int, Mat3f >& affine_maps )
{
  DBG_OUT( "Computing smooth depth + gradient" );

  cv::Mat1d ii_depth_map;
  cv::Mat_<uint32_t> ii_depth_count;
  depthIntegral( depth_map, ii_depth_map, ii_depth_count );

  if ( det_params_.affine_multiscale_ )
  {
    for (std::set<int>::iterator it = octaves.begin(); it != octaves.end(); it++ )
    {
      int octave = *it;
      double scale = det_params_.base_scale_ * std::pow( 2.0, float(octave) );

      smoothed_depth_maps[octave] = Mat1f();
      affine_maps[octave] = Mat3f();

      Mat1f& smoothed_depth_map = smoothed_depth_maps[octave];
      Mat3f& affine_map = affine_maps[octave];

      smoothDepth( scale_map, ii_depth_map, ii_depth_count, scale, smoothed_depth_map );
      computeAffineMap( scale_map, smoothed_depth_map, scale, det_params_.min_px_scale_, affine_map );

  #ifdef SHOW_DEBUG_WIN
      {
      std::stringstream s; s<<" s="<<scale;

      //imshowNorm( "smoothed_depth_map"+s.str(), smoothed_depth_map, 0 );
      std::vector<cv::Mat> affine_map_channels;
      cv::split(affine_map,affine_map_channels);

      double minv,maxv;
      int tmp;
      cv::minMaxIdx( affine_map_channels[0], &minv, &maxv, &tmp, &tmp );
      imshow( "major" + s.str(), affine_map_channels[0] / maxv );

      //imshow( "minor" + s.str(), affine_map_channels[1] / maxv );
      imshow( "minor/major" + s.str(), affine_map_channels[0] );
      imshow( "major.x" + s.str(), affine_map_channels[1]*0.5+0.5 );
      imshow( "major.y" + s.str(), affine_map_channels[2]*0.5+0.5 );
      }
  #endif
    }
  }
  else
  {
    float win_size = 15.0;
    Mat1f smoothed_depth_map;

    Mat1f fake_scale_map( scale_map.rows, scale_map.cols, 1.0f );

    smoothDepth( fake_scale_map, ii_depth_map, ii_depth_count, win_size, smoothed_depth_map );

    Mat3f affine_map;
    computeAffineMapFixed( smoothed_depth_map, win_size/2, f, affine_map );
    for (std::set<int>::iterator it = octaves.begin(); it != octaves.end(); it++ )
    {
      int octave = *it;
      smoothed_depth_maps[octave] = smoothed_depth_map;
      affine_maps[octave] = affine_map;
    }

#ifdef SHOW_DEBUG_WIN
    {
    std::stringstream s;

    //imshowNorm( "smoothed_depth_map"+s.str(), smoothed_depth_map, 0 );
    std::vector<cv::Mat> affine_map_channels;
    cv::split(affine_map,affine_map_channels);

    //imshow( "minor" + s.str(), affine_map_channels[1] / maxv );
    imshow( "smooth depth" + s.str(), smoothed_depth_map );
    imshow( "minor/major" + s.str(), affine_map_channels[0] );
    imshow( "major.x" + s.str(), affine_map_channels[1]*0.5+0.5 );
    imshow( "major.y" + s.str(), affine_map_channels[2]*0.5+0.5 );
    }
#endif
  }
}

void DAFT::computeScaleMap( const Mat1f &depth_map, const Mat1b &mask, float f, Mat1f &scale_map )
{
  Mat1f::iterator scale_it = scale_map.begin();
  Mat1f::iterator scale_map_end = scale_map.end();
  Mat1f::const_iterator depth_it = depth_map.begin();
  Mat1b::const_iterator mask_it = mask.begin();

  for (; scale_it != scale_map_end; ++scale_it, ++depth_it, ++mask_it )
  {
    if ( *mask_it != 0 && *depth_it > 0.0 )
    {
      float s = f / *depth_it;
      *scale_it = s;
    }
    else
    {
      *scale_it = std::numeric_limits<float>::quiet_NaN();
    }
  }
}

void DAFT::getOctaves( const Mat1f &scale_map, float max_px_scale, std::set<int> &pyr_octaves, std::set<int> &det_octaves )
{
  pyr_octaves.clear();
  det_octaves.clear();

  int min_octave;
  int n_octaves;

  // Compute scale map from depth map & optionally determine min/max possible octave
  if (det_params_.scale_levels_ == det_params_.AUTO)
  {
    float min_scale_fac = std::numeric_limits<float>::infinity();
    float max_scale_fac = 0;

    Mat1f::const_iterator scale_it = scale_map.begin();
    Mat1f::const_iterator scale_map_end = scale_map.end();

    for (; scale_it != scale_map_end; ++scale_it )
    {
      if ( finite(*scale_it) )
      {
        const float s = *scale_it;
        if (s > max_scale_fac)
        {
          max_scale_fac = s;
        }
        if (s < min_scale_fac)
        {
          min_scale_fac = s;
        }
      }
    }

    if (!finite(min_scale_fac) || max_scale_fac == 0)
    {
      DBG_OUT( "error: depth data invalid." );
      return;
    }

    DBG_OUT( "min_scale_fac" << min_scale_fac );
    DBG_OUT( "max_scale_fac" << max_scale_fac );

    double delta_n_min = det_params_.min_px_scale_
        / (max_scale_fac * det_params_.base_scale_);
    double delta_n_max = max_px_scale / (min_scale_fac * det_params_.base_scale_);

    double n_min = std::ceil(std::log(delta_n_min) / std::log(2.0));
    double n_max = std::floor(std::log(delta_n_max) / std::log(2.0));

    min_octave = n_min;
    n_octaves = n_max - n_min;
  }
  else
  {
    min_octave = 1;
    n_octaves = det_params_.scale_levels_;
  }

  // compute which octaves we need
  for (int octave = min_octave; octave < min_octave+n_octaves; octave++)
  {
    det_octaves.insert(octave);
    // insert octave and next bigger one (for diff-of-gaussian)
    pyr_octaves.insert(octave);
    pyr_octaves.insert(octave+1);
    pyr_octaves.insert(octave+desc_params_.octave_offset_);
    pyr_octaves.insert(octave+desc_params_.octave_offset_+1);
  }
}


bool DAFT::prepareData(const cv::Mat &image, const cv::Mat &depth_map_orig,
    Mat& gray_image, Mat1d& ii, cv::Mat1f& depth_map, cv::Mat1b& mask )
{
  gray_image = image;

  // Convert RGB to Grey image
  if (image.type() == CV_8UC3) {
    cvtColor(image, gray_image, CV_BGR2GRAY);
  }

  // Construct integral image
  switch (gray_image.type())
  {
  case CV_8U:
  {
    cv::Mat1b m_in = gray_image;
    integral2( m_in, ii, 1.0/255.0 );
  }
  break;
  case CV_64F:
  {
    cv::Mat1d m_in = gray_image;
    integral2( m_in, ii );
  }
  break;
  case CV_32F:
  {
    cv::Mat1f m_in = gray_image;
    integral2( m_in, ii );
  }
  break;
  default:
    return false;
    break;
  }

  // Convert depth map to floating point
  if (depth_map_orig.type() == CV_16U) {
    depth_map_orig.convertTo(depth_map, CV_32F, 0.001, 0.0);
  } else if (depth_map_orig.type() == CV_32F) {
    depth_map= depth_map_orig;
  }
  else
  {
    return false;
  }

#ifdef SHOW_MASK
  imshow( "mask", mask*255 );
#endif

  for ( int y=0; y<depth_map.rows; y++ )
  {
    for ( int x=0; x<depth_map.cols; x++ )
    {
      if ( isnan(depth_map(y,x)) || depth_map(y,x) == 0.0 )
      {
        mask(y,x) = 0;
      }
    }
  }

#ifdef SHOW_MASK
  imshow( "mask new", mask*255 );
#endif

#ifdef SHOW_DEBUG_WIN
  imshow( "depth_map", depth_map );
#endif

  closeGaps<50>( depth_map, depth_map, 0.5 );

#ifdef SHOW_DEBUG_WIN
  imshow( "depth_map closed", depth_map );
#endif

  return true;
}


}
}
