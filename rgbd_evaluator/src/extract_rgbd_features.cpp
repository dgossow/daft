/*
* Copyright (C) 2011 David Gossow
*/

#include "rgbd_evaluator/extract_rgbd_features.h"

#include <vector>

#include <boost/make_shared.hpp>
#include <sensor_msgs/image_encodings.h>

#include <rgbd_features/feature_detection.h>

#include <rgbd_features/math_stuff.h>

#include <rgbd_features/mean_filter.h>
#include <rgbd_features/hessian_filter.h>
#include <rgbd_features/dob_filter.h>
#include <rgbd_features/harris_filter.h>
#include <rgbd_features/good_features_to_track_filter.h>

// ROS
#include <ros/ros.h>

#include <sys/time.h>
#include <omp.h>

#include <Eigen/LU>

using std::vector;
using namespace rgbd_features;

///////////////////
static std::map< std::string, std::pair<double, double> > __timing_list;

#define TIME_MS( FUNCTION ) \
{ \
  timeval t1,t2; \
  gettimeofday(&t1, NULL); \
  FUNCTION \
  gettimeofday(&t2, NULL); \
  double elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0; \
/*  ROS_INFO( "%.4lf ms for " #FUNCTION, elapsed_time ); */ \
  \
  if ( __timing_list.find( #FUNCTION ) == __timing_list.end() ) { \
    __timing_list[ #FUNCTION ] = std::pair<double,double>(0,0); \
  } \
  __timing_list[ #FUNCTION ].second += elapsed_time; \
}

void increase_counters()
{
  std::map< std::string, std::pair<double, double> >::iterator timing_it = __timing_list.begin();
  while( timing_it != __timing_list.end() )
  {
    timing_it->second.first ++;
    timing_it++;
  }
}

void print_time()
{
  double time_total = 0;
  std::map< std::string, std::pair<double, double> >::iterator timing_it = __timing_list.begin();
  while( timing_it != __timing_list.end() )
  {
    double time_mean = timing_it->second.second/timing_it->second.first;
    ROS_INFO( "%.4lf ms mean time for %s", time_mean, timing_it->first.c_str() );
    time_total += time_mean;
    timing_it++;
  }
  ROS_INFO( "%.4lf ms mean time in total.", time_total );
}
////////////////////


namespace rgbd_evaluator
{

ExtractRgbdFeatures::ExtractRgbdFeatures ( ros::NodeHandle comm_nh, ros::NodeHandle param_nh ):
    scale_map_(0),
    f_(0),
    width_(0),
    height_(0),
    input_image_(0),
    detector_image_(0),
    detector_image_min_(-1),
    detector_image_max_(1)
{
  image_transport::ImageTransport it(param_nh);
  result_image_publisher_ = it.advertise("image_out", 1);

  dyn_conf_cb_ = boost::bind(&ExtractRgbdFeatures::dynConfigCb, this, _1, _2);
  dyn_conf_srv_.setCallback( dyn_conf_cb_ );

  ROS_INFO ( "ExtractRgbdFeatures ready." );
}


ExtractRgbdFeatures::~ExtractRgbdFeatures ()
{
  IntegralImage::DeallocateImage( scale_map_, height_ );
  IntegralImage::DeallocateImage( detector_image_, height_ );
  integral_image_.clean();
  print_time();
}


void ExtractRgbdFeatures::dynConfigCb(RgbdFeaturesConfig &config, uint32_t level)
{
  config_ = config;
  omp_set_num_threads( config.num_threads );
  if ( last_point_cloud_.get() )
  {
  	processPointCloud( last_point_cloud_ );
  }
}

void ExtractRgbdFeatures::processCameraInfo ( const sensor_msgs::CameraInfo::ConstPtr& msg )
{
	//copy camera parameters

  if ( f_ ) return;
  f_ = msg->P[0];
  ROS_INFO( "f=%f", f_ );
}

void ExtractRgbdFeatures::processPointCloud ( const sensor_msgs::PointCloud2::ConstPtr& point_cloud )
{
  last_point_cloud_ = point_cloud;
  computeFeatures( point_cloud );

  float *xyz = ( float* ) & ( point_cloud->data ) [ 0 ];
  int point_step = point_cloud->point_step / sizeof(float);
  int row_step = point_cloud->row_step / sizeof(float);

  for ( unsigned i=0; i<keypoints_.size(); i++ )
  {
    int x = keypoints_[i]._x;
    int y = keypoints_[i]._y;
    keypoints_[i]._rx = xyz[ y*row_step + x*point_step ];
    keypoints_[i]._ry = xyz[ y*row_step + x*point_step + 1 ];
    keypoints_[i]._rz = xyz[ y*row_step + x*point_step + 2 ];
  }
}

void ExtractRgbdFeatures::computeFeatures ( const sensor_msgs::PointCloud2::ConstPtr& msg )
{
  if ( !f_ ) return;

  point_cloud_ = msg;

  if ( ( width_ != (int)point_cloud_->width ) || ( height_ != (int)point_cloud_->height ) )
  {
    ROS_INFO ( "Image size has changed." );

    imageSizeChanged( );

    IntegralImage::DeallocateImage( scale_map_, height_ );
    scale_map_ = IntegralImage::AllocateImage(width_,height_);

    IntegralImage::DeallocateImage( input_image_, height_ );
    input_image_ = IntegralImage::AllocateImage(width_,height_);

    IntegralImage::DeallocateImage( detector_image_, height_ );
    detector_image_ = IntegralImage::AllocateImage(width_,height_);
  }


  switch ( config_.input )
  {
  case RgbdFeatures_channel_intensity:
    TIME_MS( fillIntensityImage( ); )
    break;
  case RgbdFeatures_channel_depth:
    TIME_MS( fillDepthImage( ); )
    break;
  case RgbdFeatures_channel_chroma:
    TIME_MS( fillChromaImage( ); )
    break;
  }

  fillIntegralImage();

  TIME_MS( calcScaleMap( ); )

  keypoints_.clear();
  keypoints_.reserve(50000);

  double scale = config_.base_scale;

  for( int scale_level = 0; scale_level < config_.scale_levels; scale_level++, scale *= config_.scale_step )
  {
    switch ( config_.detector_type )
    {
    case RgbdFeatures_det_determinant_of_hessian:
      TIME_MS( filter<HessianFilter>( integral_image_, scale_map_, scale, detector_image_ ); )
      break;
    case RgbdFeatures_det_difference_of_boxes_small:
      TIME_MS( filter< DobFilter<2>  >( integral_image_, scale_map_,
          scale, detector_image_ ); )
      break;
    case RgbdFeatures_det_difference_of_boxes_large:
      TIME_MS( filter< DobFilter<3>  >( integral_image_, scale_map_,
          scale, detector_image_ ); )
      break;
    case RgbdFeatures_det_harris_corners_3x3:
      TIME_MS( filter< HarrisFilter<1>  >( integral_image_, scale_map_,
          scale, detector_image_ ); )
      break;
    case RgbdFeatures_det_harris_corners_4x4:
      TIME_MS( filter< HarrisFilter<2>  >( integral_image_, scale_map_,
          scale, detector_image_ ); )
      break;
    case RgbdFeatures_det_good_features_to_track_3x3:
      TIME_MS( filter< GoodFeaturesToTrackFilter<1>  >( integral_image_, scale_map_,
          scale, detector_image_ ); )
      break;
    default:
      ROS_ERROR( "Invalid detector_type selected! Check dynamic_reconfigure params." );
      break;
    }

    TIME_MS( findMaxima( integral_image_, scale_map_, scale,
        detector_image_, keypoints_, config_.det_threshold ); )
  }

  ROS_INFO( "Found %i keypoints.", (int)keypoints_.size() );

  switch ( config_.postfilter_type )
  {
  case RgbdFeatures_pf_none:
    break;
  case RgbdFeatures_pf_harris_corners_3x3:
    TIME_MS ( filterMaxima< HarrisFilter<1> >( integral_image_, detector_image_,
        keypoints_, config_.pf_threshold ); )
    break;
  case RgbdFeatures_pf_harris_corners_4x4:
    TIME_MS ( filterMaxima< HarrisFilter<2> >( integral_image_, detector_image_,
        keypoints_, config_.pf_threshold ); )
    break;
  case RgbdFeatures_pf_good_features_to_track_3x3:
    TIME_MS ( filterMaxima< GoodFeaturesToTrackFilter<1>  >( integral_image_, detector_image_,
        keypoints_, config_.pf_threshold ); )
    break;
  default:
    ROS_ERROR( "Invalid detector_type selected! Check dynamic_reconfigure params." );
    break;
  }

  ROS_INFO( "%i keypoints left after post-filtering.", (int)keypoints_.size() );

  publishImage();

  increase_counters();
}


void ExtractRgbdFeatures::imageSizeChanged( )
{
	width_ = point_cloud_->width;
	height_ = point_cloud_->height;

	integral_image_.clean();
	integral_image_.init( width_, height_ );

	//init published image
  result_image_.header.frame_id = "/openni_rgb_optical_frame";
  result_image_.height = height_;
  result_image_.width = width_;

  result_image_.encoding = sensor_msgs::image_encodings::RGB8;
  result_image_.data.resize(width_* height_* 3);
  result_image_.step = width_ * 3;
}


void ExtractRgbdFeatures::fillIntensityImage( )
{
  static double norm = 1.0 / 255.0 / 3.0;

  unsigned char *rgb = ( unsigned char* ) & ( point_cloud_->data ) [16];
  int point_step = point_cloud_->point_step;

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
    	input_image_[y][x] = double( rgb[0]+rgb[1]+rgb[2] ) * norm;
      rgb += point_step;
    }
  }
}


void ExtractRgbdFeatures::fillChromaImage( )
{
  unsigned char *rgb = ( unsigned char* ) & ( point_cloud_->data ) [16];
  int point_step = point_cloud_->point_step;

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
    	double max_rg = rgb[0] > rgb[1] ? rgb[0] : rgb[1];
    	double max_gb = rgb[1] > rgb[2] ? rgb[1] : rgb[2];
    	double max_rgb = max_rg > max_gb ? max_rg : max_gb;

    	double min_rg = rgb[0] < rgb[1] ? rgb[0] : rgb[1];
    	double min_gb = rgb[1] < rgb[2] ? rgb[1] : rgb[2];
    	double min_rgb = min_rg < min_gb ? min_rg : min_gb;

    	double chroma = max_rgb - min_rgb;

    	input_image_[y][x] = chroma / 255.0;
      rgb += point_step;
    }
  }
}


void ExtractRgbdFeatures::fillDepthImage( )
{
  float *z = ( float* ) & ( point_cloud_->data ) [ 2*sizeof(float) ];
  int point_step = point_cloud_->point_step / sizeof(float);

  double depth = 1;

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      if ( !isnan(*z) ) depth = *z;
      input_image_[y][x] = depth;
      z += point_step;
    }
  }
}


void ExtractRgbdFeatures::fillIntegralImage( )
{
  //transfer input to integral image
  double **integral_img = integral_image_.getIntegralImage();

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      integral_img[y+1][x+1] = input_image_[y][x] + integral_img[y][x+1] + integral_img[y+1][x] - integral_img[y][x];
    }
  }
}


void ExtractRgbdFeatures::calcScaleMap( )
{
  float *z = ( float* ) & ( point_cloud_->data ) [ 2*sizeof(float) ];
  int point_step = point_cloud_->point_step / sizeof(float);

  //calc scale map
//#pragma omp parallel for
  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x, z += point_step )
    {
      if ( isnan(*z) )
      {
        scale_map_[y][x] = -1;
        continue;
      }

      scale_map_[y][x] = f_ / *z;
    }
  }
}

void ExtractRgbdFeatures::publishImage()
{

  if ( result_image_publisher_.getNumSubscribers() == 0 )
  {
    return;
  }

  switch ( config_.image_display )
  {
  case RgbdFeatures_detector_response:
    paintDetectorImage();
    break;
  case RgbdFeatures_input:
    paintInputImage();
    break;
  case RgbdFeatures_input_and_detector_response:
    paintInputAndDetectorImage();
    break;
  default:
    break;
  }

  if ( config_.paint_keypoints ) paintKeypoints( keypoints_, 0, 255, 0 );

  result_image_publisher_.publish (boost::make_shared<const sensor_msgs::Image> (result_image_));
}


void ExtractRgbdFeatures::paintKeypoints( std::vector< KeyPoint > &keypoints, int r, int g, int b )
{
  for ( unsigned i=0; i<keypoints.size(); i++ )
  {
    int x=keypoints[i]._x;
    int y=keypoints[i]._y;
    double radius = keypoints[i]._image_scale;

    rasterCircle( x, y, radius, r/2, g/2, b/2 );

    if ( x>0 && x<width_ && y>0 && y<height_ )
    {
      result_image_.data[(y*width_+x)*3+0] = r;
      result_image_.data[(y*width_+x)*3+1] = g;
      result_image_.data[(y*width_+x)*3+2] = b;
    }
  }
}


void ExtractRgbdFeatures::paintInputImage( )
{
	double min_val = 100000;
	double max_val = -100000;
	getMinMax( input_image_, min_val, max_val );
	double norm = 255.0 / ( max_val - min_val );

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      double l = ( input_image_[y][x] + min_val ) * norm;
      if ( scale_map_[y][x] > 0 )
      {
        result_image_.data[(y*width_+x)*3+0] = l;
        result_image_.data[(y*width_+x)*3+1] = l;
        result_image_.data[(y*width_+x)*3+2] = l;
      }
      else
      {
        result_image_.data[(y*width_+x)*3+0] = 0;
        result_image_.data[(y*width_+x)*3+1] = l*0.5;
        result_image_.data[(y*width_+x)*3+2] = l;
      }
    }
  }
}

void ExtractRgbdFeatures::paintInputAndDetectorImage( )
{
  updateDetectorLimits();

	double min_val = 100000;
	double max_val = -100000;
	getMinMax( input_image_, min_val, max_val );
	double norm = 255.0 / ( max_val - min_val );

  //copy rgb to intensity / integral image
  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      double l = ( input_image_[y][x] + min_val ) * norm;
      if ( isnan (detector_image_[y][x]) )
      {
        result_image_.data[(y*width_+x)*3+0] = 0;
        result_image_.data[(y*width_+x)*3+1] = l*0.5;
        result_image_.data[(y*width_+x)*3+2] = l;
      }
      else
      {
        float val = (detector_image_[y][x]-detector_image_min_) / (detector_image_max_-detector_image_min_);
        if ( val < 0.0 )
        {
          val = 0.0;
        }
        if ( val > 1.0 )
        {
          val = 1.0;
        }
        result_image_.data[(y*width_+x)*3+0] = (1.0-val)*l + val*255; // l-d*(l+255)
        result_image_.data[(y*width_+x)*3+1] = (1.0-val)*l;
        result_image_.data[(y*width_+x)*3+2] = (1.0-val)*l;
      }
    }
  }
}


void ExtractRgbdFeatures::getMinMax( double **image, double &min_val, double &max_val )
{
	min_val *= 0.5;
	max_val *= 0.5;

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      if ( image[y][x] == std::numeric_limits<double>::quiet_NaN() ||
      		image[y][x] == std::numeric_limits<double>::infinity() )
      {
        continue;
      }
      if ( image[y][x] > max_val )
      {
      	max_val = image[y][x];
      }
      if ( image[y][x] < min_val )
      {
      	min_val = image[y][x];
      }
    }
  }

  min_val = 0.0;
}


void ExtractRgbdFeatures::updateDetectorLimits()
{
  detector_image_max_ *= 0.5;

  getMinMax( detector_image_, detector_image_min_, detector_image_max_ );

  detector_image_min_ = 0.0;
}


void ExtractRgbdFeatures::paintDetectorImage()
{
  updateDetectorLimits();

  //ROS_INFO( "min h: %f max h: %f", minH_, maxH_ );

  for ( int y=0; y<height_; ++y )
  {
    for ( int x=0; x<width_; ++x )
    {
      if ( isnan (detector_image_[y][x]) )
      {
        result_image_.data[(y*width_+x)*3+0] = 0;
        result_image_.data[(y*width_+x)*3+1] = 128;
        result_image_.data[(y*width_+x)*3+2] = 255;
      }
      else
      {
        float val = (detector_image_[y][x]-detector_image_min_) / (detector_image_max_-detector_image_min_);
        val *= 5;
        if ( val < 0.0 )
        {
          val = 0.0;
        }
        if ( val > 1.0 )
        {
          val = 1.0;
        }
        val = 1.0 - val;
        result_image_.data[(y*width_+x)*3+0] = val * 255;
        result_image_.data[(y*width_+x)*3+1] = val * 255;
        result_image_.data[(y*width_+x)*3+2] = val * 255;
      }
    }
  }
}


inline void ExtractRgbdFeatures::setPixel(int x, int y, int r, int g, int b)
{
  if ( config_.thick_lines )
  {
    if ( x>1 && x<width_-1 && y>1 && y<height_-1 )
    {
      for ( int y2 = y-1; y2<=y+1; y2++ )
      {
        for ( int x2 = x-1; x2<=x+1; x2++ )
        {
          result_image_.data[(y2*width_+x2)*3+0] = result_image_.data[(y2*width_+x2)*3+0] / 2 + r;
          result_image_.data[(y2*width_+x2)*3+1] = result_image_.data[(y2*width_+x2)*3+1] / 2 + g;
          result_image_.data[(y2*width_+x2)*3+2] = result_image_.data[(y2*width_+x2)*3+2] / 2 + b;
        }
      }
    }
  }
  else
  {
    if ( x>0 && x<width_ && y>0 && y<height_ )
    {
      result_image_.data[(y*width_+x)*3+0] = result_image_.data[(y*width_+x)*3+0] / 2 + r;
      result_image_.data[(y*width_+x)*3+1] = result_image_.data[(y*width_+x)*3+1] / 2 + g;
      result_image_.data[(y*width_+x)*3+2] = result_image_.data[(y*width_+x)*3+2] / 2 + b;
    }
  }
}


void ExtractRgbdFeatures::rasterCircle(int x0, int y0, int radius, int r, int g, int b )
{
  int f = 1 - radius;
  int ddF_x = 1;
  int ddF_y = -2 * radius;
  int x = 0;
  int y = radius;

  setPixel(x0, y0 + radius, r, g, b);
  setPixel(x0, y0 - radius, r, g, b);
  setPixel(x0 + radius, y0, r, g, b);
  setPixel(x0 - radius, y0, r, g, b);

  while(x < y)
  {
    // ddF_x == 2 * x + 1;
    // ddF_y == -2 * y;
    // f == x*x + y*y - radius*radius + 2*x - y + 1;
    if(f >= 0)
    {
      y--;
      ddF_y += 2;
      f += ddF_y;
    }
    x++;
    ddF_x += 2;
    f += ddF_x;
    setPixel(x0 + x, y0 + y, r, g, b);
    setPixel(x0 - x, y0 + y, r, g, b);
    setPixel(x0 + x, y0 - y, r, g, b);
    setPixel(x0 - x, y0 - y, r, g, b);
    setPixel(x0 + y, y0 + x, r, g, b);
    setPixel(x0 - y, y0 + x, r, g, b);
    setPixel(x0 + y, y0 - x, r, g, b);
    setPixel(x0 - y, y0 - x, r, g, b);
  }
}

}

