/*
* Copyright (C) 2011 David Gossow
*/

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <rgbd_features_cv/rgbd_cv.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo > RgbdSyncPolicy;

cv::RgbdFeatures rgbd_features;

cv_bridge::CvImageConstPtr orig_rgb_image;
cv_bridge::CvImageConstPtr orig_depth_image;
cv::Mat depth_image, rgb_image;

void rgbdImageCb(const sensor_msgs::Image::ConstPtr ros_rgb_image,
          const sensor_msgs::Image::ConstPtr ros_depth_image,
          const sensor_msgs::CameraInfo::ConstPtr ros_camera_info )
{
  orig_rgb_image = cv_bridge::toCvCopy( ros_rgb_image, "mono8" );
  orig_depth_image = cv_bridge::toCvCopy( ros_depth_image, sensor_msgs::image_encodings::TYPE_32FC1 );

  int scale_fac = orig_rgb_image->image.cols / orig_depth_image->image.cols;

  // Resize depth to have the same width as rgb
  cv::resize( orig_depth_image->image, depth_image, cvSize(0,0), scale_fac, scale_fac, cv::INTER_LINEAR );

  // Crop rgb so it has the same size as depth
  rgb_image = cv::Mat( orig_rgb_image->image, cv::Rect( 0,0, depth_image.cols, depth_image.rows ) );

  assert( depth_image.cols == rgb_image.cols && depth_image.rows == rgb_image.rows );

  cv::Matx33d camera_matrix( ros_camera_info->P.data() );

  cv::imshow( "RGB", rgb_image );
  cv::imshow( "Depth", depth_image );

  cv::Mat1f border_image;
  cv::GaussianBlur( rgb_image, border_image, cv::Size(21,21), 2, 2 );
  cv::Laplacian( border_image, border_image, CV_32FC1, 3, 10.0/255.0 );

  cv::Mat1f depth_border_image;
  cv::GaussianBlur( depth_image, depth_border_image, cv::Size(21,21), 1, 1 );
  cv::Laplacian( depth_border_image, depth_border_image, CV_32FC1, 3, 30 );

  // filter nans
  cv::Mat1f::iterator db_it = depth_border_image.begin(), db_end = depth_border_image.end();
  for (; db_it != db_end ; ++db_it)
  {
    if (isnan(*db_it)) *db_it=0;
  }

  cv::Mat1b hyper_border_image;

  //cv::multiply( border_image, depth_border_image, hyper_border_image, 10.0, CV_32FC1 );
  cv::add( border_image, depth_border_image, hyper_border_image );

  cv::imshow( "Borders", border_image );
  cv::imshow( "Depth Borders", depth_border_image );
  cv::imshow( "Hyper Borders", hyper_border_image );

  cv::waitKey(3);
}



int main( int argc, char** argv )
{
  ros::init( argc, argv, "find_tracks_node" );

  ros::NodeHandle comm_nh(""); // for topics, services

  message_filters::Subscriber<sensor_msgs::Image> rgb_img_sub(comm_nh, "rgb_image", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_img_sub(comm_nh, "depth_image", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub(comm_nh, "camera_info", 1);

  message_filters::Synchronizer<RgbdSyncPolicy> rgbd_sync(
      RgbdSyncPolicy(10), rgb_img_sub, depth_img_sub, cam_info_sub );
  rgbd_sync.registerCallback( &rgbdImageCb );

  ROS_INFO ("Subscribed to RGB image on: %s", rgb_img_sub.getTopic().c_str ());
  ROS_INFO ("Subscribed to depth image on: %s", depth_img_sub.getTopic().c_str ());
  ROS_INFO ("Subscribed to camera info on: %s", cam_info_sub.getTopic().c_str ());

  //cv::namedWindow("RGB");
  //cv::namedWindow("Depth");

  ros::spin();
}
