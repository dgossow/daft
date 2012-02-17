/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include "rgbd_features_cv/daft.h"
#include "rgbd_features_cv/filters.h"
#include "rgbd_features_cv/feature_detection.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/timer.hpp>

namespace cv
{

DAFT::DAFT(const DetectorParams & detector_params) :
    detector_params_(detector_params)
{

}

DAFT::~DAFT()
{

}

void DAFT::detect(const cv::Mat &image, const cv::Mat &depth_map_orig, cv::Matx33f camera_matrix,
    std::vector<KeyPoint3D> & kp)
{
  if ( image.size != depth_map_orig.size )
  {
    return;
  }

  Mat gray_image_orig = image;

  // Convert RGB to Grey image
  if ( image.type() == CV_8UC3 )
  {
    cvtColor( image, gray_image_orig, CV_BGR2GRAY );
  }

  // Convert 8-Bit to Float image
  Mat gray_image = gray_image_orig;
  if ( gray_image.type() != CV_64F )
  {
    gray_image_orig.convertTo( gray_image, CV_64F, 1.0/255.0, 0.0 );
  }

  // Convert depth map to floating point
  cv::Mat1f depth_map;
  if ( depth_map.type() == CV_16U )
  {
    depth_map_orig.convertTo( depth_map, CV_32F, 0.001, 0.0 );
  }
  else
  {
    depth_map = depth_map_orig;
  }

  // Compute scale map from depth map
  const float f = camera_matrix(0,0);

  Mat1d scale_map( gray_image_orig.rows, gray_image_orig.cols );
  Mat1d::iterator scale_it = scale_map.begin();
  Mat1d::iterator scale_map_end = scale_map.end();
  Mat1f::iterator depth_it = depth_map.begin();

  for (; scale_it != scale_map_end ; ++scale_it, ++depth_it)
  {
    if ( isnan(*depth_it) )
    {
      *scale_it = -1.0f;
    }
    else
    {
      *scale_it = f / *depth_it;
    }
  }

  // Construct integral image
  Mat1d ii;
  integral( gray_image, ii );

  kp.clear();
  kp.reserve(50000);

  Mat1d detector_image;
  detector_image.create( gray_image_orig.rows, gray_image_orig.cols );

  double scale = detector_params_.base_scale_;

  // detect keypoints
  for( unsigned scale_level = 0; scale_level < detector_params_.scale_levels_; scale_level++, scale *= detector_params_.scale_step_ )
  {
    // compute filter response for all pixels
    switch ( detector_params_.det_type_ )
    {
    case DetectorParams::DET_DOB:
      convolve<dob>( ii, scale_map, scale, detector_image );
      break;
    case DetectorParams::DET_LAPLACE:
      convolve<laplace>( ii, scale_map, scale, detector_image );
      break;
    case DetectorParams::DET_HARRIS:
      convolve<harris>( ii, scale_map, scale, detector_image );
      break;
    case DetectorParams::DET_DOBP:
      convolveAffine<dobAffine>( ii, scale_map, depth_map, camera_matrix, scale, detector_image );
      break;
    default:
      return;
    }

    // save index where new kps will be inserted
    unsigned kp_first = kp.size();

    // find maxima in response
    switch ( detector_params_.max_search_algo_ )
    {
    case DetectorParams::MAX_WINDOW:
      findMaxima( detector_image, scale_map, scale, detector_params_.det_threshold_, kp );
      break;
    case DetectorParams::MAX_FAST:
      findMaximaMipMap( detector_image, scale_map, scale, detector_params_.det_threshold_, kp );
      break;
    case DetectorParams::MAX_EVAL:
      {
      boost::timer timer;
      timer.restart();
      for ( int i=0; i<100; i++ )
      {
        kp.clear();
        kp.reserve(50000);
        findMaximaMipMap( detector_image, scale_map, scale, detector_params_.det_threshold_, kp );
      }
      std::cout << "findMaximaMipMap execution time [ms]: " << timer.elapsed()*10 << std::endl;
      timer.restart();
      for ( int i=0; i<100; i++ )
      {
        kp.clear();
        kp.reserve(50000);
        findMaxima( detector_image, scale_map, scale, detector_params_.det_threshold_, kp );
      }
      std::cout << "findMaxima execution time [ms]: " << timer.elapsed()*10 << std::endl;
      }
      break;
    default:
      return;
    }

    // filter found maxima by applying a threshold on a second kernel
    switch ( detector_params_.pf_type_ )
    {
    case DetectorParams::PF_NONE:
      break;
    case DetectorParams::PF_HARRIS:
      filterKeypoints<harris>( ii, detector_params_.pf_threshold_, kp );
      break;
    default:
      return;
    }

    float f_inv = 1.0 / f;
    float cx = camera_matrix(0,2);
    float cy = camera_matrix(1,2);
    for ( unsigned k=kp_first; k<kp.size(); k++ )
    {
      int kp_x = kp[k].pt.x;
      int kp_y = kp[k].pt.y;
      pt3d( f_inv, cx, cy, kp_x, kp_y, depth_map[kp_y][kp_x], kp[k].pt3d );
      //todo: normal
      getAffine( ii, depth_map, kp_x, kp_y, kp[k].size / 2, scale, kp[k].affine_mat );
    }

  }


#if 1
  {
  cv::Mat display_image;
  detector_image.convertTo( display_image, CV_8UC1, 512.0, 0.0 );

  cv::drawKeypoints3D( display_image, kp, display_image, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  std::ostringstream s;
  s << "Detector type=" << detector_params_.det_type_;
  cv::imshow( s.str(), display_image );

  /*
  cv::imshow( "dep orig", depth_map_orig );
  cv::imshow( "dep", depth_map );
  cv::imshow( "det", detector_image );
  cv::imshow( "scale", scale_map );
  */
  }
#endif
}

}
