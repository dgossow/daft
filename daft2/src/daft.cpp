/*
 * DAFT
 * Copyright (C) 2011 David Gossow
 */

#include "daft.h"
#include "filter_kernels.h"
#include "depth_filter.h"
#include "feature_detection.h"
#include "descriptor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/timer.hpp>

#include <cmath>
#include <set>
#include <map>

namespace cv {
namespace daft2 {

boost::timer t;

#define TIMER_START t.restart();
#define TIMER_STOP std::cout << "++++++ Time elapsed: " << t.elapsed()*1000.0 << "ms ++++++" << std::endl;

//DEBUG FLAGS.
//
//#define SHOW_DEBUG_WIN
//#define FIND_MAXKP

DAFT::DAFT(const DetectorParams & detector_params, const DescriptorParams & descriptor_params ) :
    det_params_(detector_params), desc_params_(descriptor_params) {

}

DAFT::~DAFT() {
}


void DAFT::operator()(const cv::Mat &image, const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp )
{
  std::cout << "Preprocessing" << std::endl;

  TIMER_START

  if (image.size != depth_map_orig.size) {
    return;
  }

  // Convert input to correct format
  Mat gray_image;
  Mat1d ii;
  Mat1f depth_map;
  prepareData( image, depth_map_orig, gray_image, ii, depth_map );

  // Initialize parameters
  int n_octaves = det_params_.scale_levels_;

  float max_px_scale = 4.0 * std::min(image.rows, image.cols) / desc_params_.patch_size_ / pow(2.0,desc_params_.octave_offset_);
  if ( det_params_.max_px_scale_ != det_params_.AUTO )
  {
    max_px_scale = std::min( max_px_scale, det_params_.max_px_scale_ );
  }

  const float f = K(0, 0);

  // Integrate Depth + Depth count (#of valid pixels)
  cv::Mat1d ii_depth_map;
  cv::Mat_<uint32_t> ii_depth_count;
  depthIntegral( depth_map, ii_depth_map, ii_depth_count );

#ifdef SHOW_DEBUG_WIN
  imshowNorm("ii_depth_map",ii_depth_map,0);
  //imshowNorm("ii_depth_count",ii_depth_count,0);
#endif

  // Compute scale map from depth map
  Mat1f scale_map(gray_image.rows, gray_image.cols);
  Mat1f::iterator scale_it = scale_map.begin();
  Mat1f::iterator scale_map_end = scale_map.end();
  Mat1f::iterator depth_it = depth_map.begin();

  int min_octave = 0;

  if (det_params_.scale_levels_ == det_params_.AUTO)
  {
    float min_scale_fac = std::numeric_limits<float>::infinity();
    float max_scale_fac = 0;

    for (; scale_it != scale_map_end; ++scale_it, ++depth_it)
    {
      if (finite(*depth_it))
      {
        float s = f / *depth_it;
        *scale_it = s;
        if (s > max_scale_fac)
          max_scale_fac = s;
        if (s < min_scale_fac)
          min_scale_fac = s;
      } else {
        *scale_it = std::numeric_limits<float>::quiet_NaN();
      }
    }

    if (!finite(min_scale_fac) || max_scale_fac == 0)
    {
      std::cout << "error: depth data invalid." << std::endl;
      return;
    }

    double delta_n_min = det_params_.min_px_scale_
        / (max_scale_fac * det_params_.base_scale_);
    double delta_n_max = max_px_scale / (min_scale_fac * det_params_.base_scale_);

    double n_min = std::ceil(log(delta_n_min) / log(2.0));
    double n_max = std::floor(log(delta_n_max) / log(2.0));

    min_octave = n_min;
    n_octaves = n_max - n_min + 1;
  }
  else
  {
    for (; scale_it != scale_map_end; ++scale_it, ++depth_it)
    {
      *scale_it = f / *depth_it;
    }
  }

  kp.clear();
  kp.reserve(50000);

  // make list of all octaves that we need to compute
  std::set<int> octaves;

  // compute which octaves we need
  for (int octave = min_octave; octave < min_octave+n_octaves; octave++)
  {
  	octaves.insert(octave);
  	octaves.insert(octave+1);
  	octaves.insert(octave+desc_params_.octave_offset_);
  	octaves.insert(octave+desc_params_.octave_offset_+1);
  }

  std::map< int, Mat1f > smoothed_imgs;
  std::map< int, Mat1f> smoothed_depth_maps;
  std::map< int, Mat3f > affine_maps;

  // compute depth-normalized image pyramid

  TIMER_STOP

  std::cout << "Computing smooth depth + gradient" << std::endl;

  TIMER_START
  for (std::set<int>::iterator it = octaves.begin(); it != octaves.end(); it++ )
  {
  	int octave = *it;
  	double scale = det_params_.base_scale_ * std::pow( 2.0, float(octave) );

    smoothed_imgs[octave] = Mat1f();
    smoothed_depth_maps[octave] = Mat1f();
    affine_maps[octave] = Mat3f();

    Mat1f& smoothed_depth_map = smoothed_depth_maps[octave];
    Mat3f& affine_map = affine_maps[octave];

    smoothDepth( scale_map, ii_depth_map, ii_depth_count, scale, smoothed_depth_map );
    computeAffineMap( scale_map, smoothed_depth_map, scale, det_params_.min_px_scale_, affine_map );
  }
  TIMER_STOP

  std::cout << "Computing gaussians" << std::endl;

  TIMER_START
  for (std::set<int>::iterator it = octaves.begin(); it != octaves.end(); it++ )
  {
    int octave = *it;
    double scale = det_params_.base_scale_ * std::pow( 2.0, float(octave) );
    Mat1f& smoothed_img = smoothed_imgs[octave];
    Mat1f& smoothed_depth_map = smoothed_depth_maps[octave];
    Mat3f& affine_map = affine_maps[octave];

#ifdef SHOW_DEBUG_WIN
    {
    std::stringstream s; s<<" s="<<scale;

    imshowNorm( "smoothed_depth_map"+s.str(), smoothed_depth_map, 0 );
    std::vector<cv::Mat> affine_map_channels;
    cv::split(affine_map,affine_map_channels);
    imshowNorm( "major.x" + s.str(), affine_map_channels[0]*0.5+0.5, 0 );
    imshowNorm( "major.y" + s.str(), affine_map_channels[1]*0.5+0.5, 0 );
    imshowNorm( "minor_ratio" + s.str(), affine_map_channels[2], 0 );
    }
#endif

    // compute filter response for all pixels
    switch (det_params_.det_type_) {
    case DetectorParams::DET_FELINE:
      if (det_params_.affine_) {
        convolveAffine<felineAffine>(ii, scale_map, affine_map,
            scale, 1, smoothed_img );
      } else {
        convolve<boxMean>(ii, scale_map, scale, 1, smoothed_img);
      }
      break;
    default:
      std::cout << "error: invalid detector type: " << det_params_.det_type_ << std::endl;
      return;
    }

#ifdef SHOW_DEBUG_WIN
    {
      static int i=0;
      cv::Mat display_image;
      smoothed_img.convertTo( display_image, CV_8UC1, 255, 0.0 );
      std::ostringstream s;
      s << "frame # " << i;
      //cv::putText( display_image, s.str( ), Point(10,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0) );

      s.str("");
      s << "Smooth Detector type=" << det_params_.det_type_ << " max=" << det_params_.max_search_algo_ << " scale = " << scale << " affine = " << det_params_.affine_;
      imshow( s.str(), cv::Mat( display_image * (1.0f/0.73469f) ) );

      /*
       cv::imshow( "dep orig", depth_map_orig );
       cv::imshow( "dep", depth_map );
       cv::imshow( "det", detector_image );
       cv::imshow( "scale", scale_map );
       */

      i++;
    }
#endif
  }
  TIMER_STOP

  Mat1f response_map;
  response_map.create(gray_image.rows, gray_image.cols);

  std::cout << "Computing response & finding max" << std::endl;

  TIMER_START
  // compute abs. difference of gaussians and detect maxima
  for (int octave = min_octave; octave < min_octave+n_octaves; octave++)
  {
  	double scale = det_params_.base_scale_ * std::pow( det_params_.scale_step_, float(octave) );

  	cv::absdiff( smoothed_imgs[octave+1], smoothed_imgs[octave], response_map );
    Mat3f& affine_map = affine_maps[octave];

    // save index where new kps will be inserted
    unsigned kp_first = kp.size();

    // find maxima in response
    switch (det_params_.max_search_algo_) {
    case DetectorParams::MAX_WINDOW:
      if ( det_params_.affine_ ) {
        findMaximaAffine(response_map, scale_map, affine_map,
            scale, det_params_.min_px_scale_, max_px_scale,
            det_params_.det_threshold_, kp);
      }
      else {
        findMaxima(response_map, scale_map, scale,
            det_params_.min_px_scale_, max_px_scale,
            det_params_.det_threshold_, kp);
      }
      break;
    case DetectorParams::MAX_FAST:
      findMaximaMipMap(response_map, scale_map, scale,
          det_params_.min_px_scale_, max_px_scale,
          det_params_.det_threshold_, kp);
      break;
    default:
      std::cout << "error: invalid max search type: " << det_params_.max_search_algo_ << std::endl;
      return;
    }

    std::cout << "octave " << octave << " scale " << scale << ": ";
    std::cout << kp.size()-kp_first << " keypoints found." << std::endl;

    // assign octave
    for ( unsigned k=kp_first; k<kp.size(); k++ )
    {
      kp[k].octave = octave;
    }

#ifdef SHOW_DEBUG_WIN
    {
      static int i=0;
      cv::Mat display_image;
      response_map.convertTo( display_image, CV_8UC1, 900, 0.0 );

      std::vector<KeyPoint3D> kp2;
      for ( unsigned k=kp_first; k<kp.size(); k++ )
      {
        KeyPoint3D kp_curr = kp[k];
        int kp_x = kp[k].pt.x;
        int kp_y = kp[k].pt.y;

        Vec3f affine_params = affine_maps[kp[k].octave](kp_y,kp_x);

        if ( affine_params[2] > 0.1 )
        {
          // keypoint shall cover outer and inner and wants size not radius
          kp_curr.affine_major = kp[k].size;
          kp_curr.affine_minor = kp[k].size * affine_params[2];
          kp_curr.affine_angle = atan2( affine_params[1], affine_params[0] );

          kp2.push_back(kp_curr);
        }
      }

      cv::drawKeypoints3D( display_image, kp2, display_image, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      std::ostringstream s;
      s << "frame # " << i;
      cv::putText( display_image, s.str( ), Point(10,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0) );

      s.str("");
      s << "Response Detector type=" << det_params_.det_type_ << " max=" << det_params_.max_search_algo_ << " scale = " << scale << " affine = " << det_params_.affine_;
      imshow( s.str(), cv::Mat( display_image * (1.0f/0.73469f) ) );

      /*
       cv::imshow( "dep orig", depth_map_orig );
       cv::imshow( "dep", depth_map );
       cv::imshow( "det", detector_image );
       cv::imshow( "scale", scale_map );
       */

      i++;
    }
#endif
  }
  TIMER_STOP

  std::cout << kp.size() << " keypoints found in first stage." << std::endl;

  TIMER_START
  // filter found maxima by applying a threshold on a second kernel
  switch (det_params_.pf_type_) {
  case DetectorParams::PF_NONE:
    break;
  case DetectorParams::PF_NEIGHBOURS:
    filterKpNeighbours(response_map, det_params_.pf_threshold_, kp);
    break;
  case DetectorParams::PF_PRINC_CURV_RATIO: {
    float r = det_params_.pf_threshold_;
    float r_thresh = (r + 1) * (r + 1) / r;
    if (det_params_.affine_) {
      filterKpKernelAffine<princCurvRatioAffine>(ii, r_thresh, kp);
    } else {
      filterKpKernel<princCurvRatio>(ii, r_thresh, kp);
#if 0
      showBig( 128, 3.0f*sDxxKernel.asCvImage() + 0.5f, "dxx" );
      showBig( 128, 3.0f*sDyyKernel.asCvImage() + 0.5f, "dyy" );
      showBig( 128, 3.0f*sDxyKernel.asCvImage() + 0.5f, "dxy" );
#endif
    }
    break;
  }

  default:
    return;
  }

  std::cout << kp.size() << " keypoints left after post-filter." << std::endl;
  TIMER_STOP
  TIMER_START

  // assign 3d points, normals and local affine params
  float f_inv = 1.0 / f;
  float cx = K(0, 2);
  float cy = K(1, 2);
  std::vector<KeyPoint3D> kp2;
  kp2.reserve(kp.size());

  // correct keypoint size according to given octave offset
  float scale_fac = pow( 2.0, float(desc_params_.octave_offset_) );

  for (unsigned k = 0; k < kp.size(); k++) {

  	kp[k].size *= scale_fac;
  	kp[k].world_size *= scale_fac;
  	kp[k].octave += desc_params_.octave_offset_;

    int kp_x = kp[k].pt.x;
    int kp_y = kp[k].pt.y;
    getPt3d(f_inv, cx, cy, kp_x, kp_y, depth_map[kp_y][kp_x], kp[k].pt3d);

    Vec3f affine_params = affine_maps[kp[k].octave](kp_y,kp_x);

    if ( affine_params[2] > 0.1 )
    {
      // keypoint shall cover outer and inner and wants size not radius
      kp[k].affine_major = kp[k].size;
      kp[k].affine_minor = kp[k].size * affine_params[2];
      kp[k].affine_angle = atan2( affine_params[1], affine_params[0] );

      // compute exact normal using pca
      kp[k].normal = getNormal(kp[k], depth_map, K, 2.0 / scale_fac );

      Mat1f& smoothed_img = smoothed_imgs[kp[k].octave];
      Mat1f& smoothed_img2 = smoothed_imgs[kp[k].octave+1];
      if ( getDesc( desc_params_.patch_size_, smoothed_img, smoothed_img2, kp[k], depth_map, K ) )
      {
        kp2.push_back(kp[k]);
      }
    }
  }
  kp.swap(kp2);

  std::cout << kp.size() << " keypoints left after descriptor computation." << std::endl;
  TIMER_STOP


#ifdef FIND_MAXKP
  cv::Mat1f patch1, patch2;

  cv::Mat display_image;
  cv::drawKeypoints3D( image, kp, display_image, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  // draw patch for strongest keypoint
  float resp_max=0;
  unsigned k_max=0;
  for ( unsigned k=0; k<kp.size(); k++ )
  {
    if ( kp[k].response > resp_max )
    {
      k_max = k;
      resp_max = kp[k].response;
    }
  }

  if ( kp.size() > 0 )
  {
    vector<KeyPoint3D> kp2;
    kp2.push_back(kp[k_max]);
    cv::Mat tmp;
    response_map.convertTo( tmp, CV_8UC1, 900, 0.0 );
    cv::drawKeypoints3D( tmp, kp2, tmp, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow("max_kp",tmp);

    int k=k_max;
    std::stringstream s;
    s << "max response patch";

    Mat1f& smoothed_img = smoothed_imgs[kp[k].octave];
    Mat1f& smoothed_img2 = smoothed_imgs[kp[k].octave+1];
    Mat1f& smoothed_depth_map = smoothed_depth_maps[kp[k].octave];

    getDesc( desc_params_.patch_size_, smoothed_img, smoothed_img2, kp[k], smoothed_depth_map, K, true );
  }

  imshow( "rgb", display_image );
#endif

}


bool DAFT::prepareData(const cv::Mat &image, const cv::Mat &depth_map_orig,
    Mat& gray_image, Mat1d& ii, cv::Mat1f& depth_map )
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
    depth_map = depth_map_orig;
  }
  else
  {
    return false;
  }

  return true;
}


}
}
