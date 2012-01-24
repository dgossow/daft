/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <rgbd_features_cv/rgbd_cv.h>
#include <rgbd_features_cv/filters.h>
#include <rgbd_features_cv/feature_detection.h>

#include <opencv2/imgproc/imgproc.hpp>

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
  if ( gray_image_float.type() != CV_32S )
  {
    gray_image.convertTo( gray_image_float, CV_32S, 1.0/255.0, 0.0 );
  }

  // Construct integral image for fast smoothing (box filter)
  Mat1d integral_image;
  integral( gray_image, integral_image, CV_32S);

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
      for (; scale_it != scale_map_end ; ++scale_it, ++depth_it)
      {
        *scale_it = *depth_it ? f / float(*depth_it) : 0.0f;
      }
    }
    break;

    case CV_32F:
    {
      Mat1d::iterator scale_it = scale_map.begin(), scale_map_end = scale_map.end();
      MatConstIterator_<float> depth_it = depth_map.begin<float>();
      for (; scale_it != scale_map_end ; ++scale_it, ++depth_it)
      {
        *scale_it = *depth_it ? f / *depth_it : 0.0f;
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

  for( unsigned scale_level = 0; scale_level < detector_params_.scale_levels_; scale_level++, scale *= detector_params_.scale_step_ )
  {
    switch ( detector_params_.detector_type_ )
    {
    case DetectorParams::DET_DOB:
      filterImage<dob>( integral_image, scale_map, scale, detector_image );
      break;
    default:
      return;
    }

    findMaxima( detector_image, scale_map, scale, keypoints, detector_params_.det_threshold_ );
  }
}


}
