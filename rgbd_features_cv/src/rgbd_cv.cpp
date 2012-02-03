/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <rgbd_features_cv/rgbd_cv.h>
#include <rgbd_features_cv/filters.h>
#include <rgbd_features_cv/feature_detection.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/timer.hpp>

namespace cv
{

RgbdFeatures::RgbdFeatures(const DetectorParams & detector_params) :
    detector_params_(detector_params)
{

}

RgbdFeatures::~RgbdFeatures()
{

}

void RgbdFeatures::detect(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f camera_matrix,
    std::vector<cv::KeyPoint> & keypoints)
{
  if ( image.size != depth_map.size )
  {
    return;
  }

  // Make sure we have a 32-bit float grey value image
  Mat gray_image = image;

  // Convert RGB to Grey
  if ( image.type() == CV_8UC3 )
  {
    cvtColor( image, gray_image, CV_BGR2GRAY );
  }

  // Convert 8-Bit to Float
  Mat gray_image_float = gray_image;
  if ( gray_image_float.type() != CV_64F )
  {
    gray_image.convertTo( gray_image_float, CV_64F, 1.0/255.0, 0.0 );
  }

  Mat1d::iterator it = gray_image_float.begin<double>();
  double maxv=0;
  for (; it != gray_image_float.end<double>(); ++it)
  {
    maxv = std::max( *it, maxv );
  }

  //std::cout << "max: " << maxv << std::endl;

  //cv::imshow("gray",gray_image_float);

  // Construct integral image for fast smoothing (box filter)
  Mat1d integral_image;
  integral( gray_image_float, integral_image);

  assert( integral_image.rows == gray_image.rows+1 && integral_image.cols == gray_image.cols+1 );

  // Compute scale map from depth map
  float f = camera_matrix(0,0);
  Mat1d scale_map( gray_image.rows, gray_image.cols );

  switch ( depth_map.type() )
  {
    case CV_16U:
    {
      f *= 1000.0;
      Mat1d::iterator scale_it = scale_map.begin(), scale_map_end = scale_map.end();
      MatConstIterator_<uint16_t> depth_it = depth_map.begin<uint16_t>();
      std::cout << *depth_it << std::endl;
      for (; scale_it != scale_map_end ; ++scale_it, ++depth_it)
      {
        *scale_it = *depth_it != (uint16_t)(-1) ? f / float(*depth_it) : 1.0f;
      }
    }
    break;

    case CV_32F:
    {
      Mat1d::iterator scale_it = scale_map.begin(), scale_map_end = scale_map.end();
      MatConstIterator_<float> depth_it = depth_map.begin<float>();
      for (; scale_it != scale_map_end ; ++scale_it, ++depth_it)
      {
        *scale_it = !isnan(*depth_it) ? f / *depth_it : -1.0f;
      }
    }
    break;

    default:
      return;
  }

  keypoints.clear();
  keypoints.reserve(50000);

  double scale = detector_params_.base_scale_;
  Mat1d detector_image;
  detector_image.create( gray_image.rows, gray_image.cols );

  // detect keypoints
  for( unsigned scale_level = 0; scale_level < detector_params_.scale_levels_; scale_level++, scale *= detector_params_.scale_step_ )
  {
    // compute filter response for all pixels
    switch ( detector_params_.det_type_ )
    {
    case DetectorParams::DET_DOB:
      filterImage<dob>( integral_image, scale_map, scale, detector_image );
      break;
    default:
      return;
    }

    // find maxima in response
    switch ( detector_params_.max_search_algo_ )
    {
    case DetectorParams::MAX_FAST:
      findMaximaMipMap( detector_image, scale_map, scale, detector_params_.det_threshold_, keypoints );
      break;
    case DetectorParams::MAX_EVAL:
      {
      boost::timer timer;
      timer.restart();
      for ( int i=0; i<100; i++ )
      {
        keypoints.clear();
        keypoints.reserve(50000);
        findMaximaMipMap( detector_image, scale_map, scale, detector_params_.det_threshold_, keypoints );
      }
      std::cout << "findMaximaMipMap execution time [ms]: " << timer.elapsed()*10 << std::endl;
      timer.restart();
      for ( int i=0; i<100; i++ )
      {
        keypoints.clear();
        keypoints.reserve(50000);
        findMaxima( detector_image, scale_map, scale, detector_params_.det_threshold_, keypoints );
      }
      std::cout << "findMaxima execution time [ms]: " << timer.elapsed()*10 << std::endl;
      }
      break;
    default:
      findMaxima( detector_image, scale_map, scale, detector_params_.det_threshold_, keypoints );
      return;
    }

    // filter found maxima by applying a threshold on a second kernel
    switch ( detector_params_.pf_type_ )
    {
    case DetectorParams::PF_NONE:
      break;
    case DetectorParams::PF_HARRIS:
      filterKeypoints<harris>( integral_image, detector_params_.pf_threshold_, keypoints );
      break;
    default:
      return;
    }

  }
}


}
