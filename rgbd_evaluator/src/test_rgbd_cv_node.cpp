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

cv_bridge::CvImageConstPtr orig_intensity_image;
cv_bridge::CvImageConstPtr orig_depth_image;
cv::Mat depth_image, intensity_image;

void rgbdImageCb(const sensor_msgs::Image::ConstPtr ros_intensity_image,
          const sensor_msgs::Image::ConstPtr ros_depth_image,
          const sensor_msgs::CameraInfo::ConstPtr ros_camera_info )
{
  ROS_INFO_STREAM( "Image received." );

  orig_intensity_image = cv_bridge::toCvCopy( ros_intensity_image, "mono8" );
  orig_depth_image = cv_bridge::toCvCopy( ros_depth_image, sensor_msgs::image_encodings::TYPE_32FC1 );

  int scale_fac = orig_intensity_image->image.cols / orig_depth_image->image.cols;

  // Resize depth to have the same width as rgb
  cv::resize( orig_depth_image->image, depth_image, cvSize(0,0), scale_fac, scale_fac, cv::INTER_NEAREST );

  // Crop rgb so it has the same size as depth
  intensity_image = cv::Mat( orig_intensity_image->image, cv::Rect( 0,0, depth_image.cols, depth_image.rows ) );

  assert( depth_image.cols == intensity_image.cols && depth_image.rows == intensity_image.rows );

  cv::Matx33d camera_matrix( ros_camera_info->P.data() );

  std::vector<cv::KeyPoint> keypoints;

  rgbd_features.detect( intensity_image, depth_image, camera_matrix, keypoints);

  ROS_INFO_STREAM( keypoints.size() << " keypoints detected." );

  cv::Scalar kp_col(0,255,0);
  cv::drawKeypoints( intensity_image, keypoints, intensity_image, kp_col, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  cv::imshow( "RGB", intensity_image );
  //cv::imshow( "Depth", depth_image );

  cv::waitKey(3);
}



int main( int argc, char** argv )
{
  ros::init( argc, argv, "rgbd_evaluator_node" );

  ros::NodeHandle comm_nh(""); // for topics, services

  message_filters::Subscriber<sensor_msgs::Image> intensity_img_sub(comm_nh, "intensity_image", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_img_sub(comm_nh, "depth_image", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub(comm_nh, "camera_info", 1);

  message_filters::Synchronizer<RgbdSyncPolicy> rgbd_sync(
      RgbdSyncPolicy(10), intensity_img_sub, depth_img_sub, cam_info_sub );
  rgbd_sync.registerCallback( &rgbdImageCb );

  ROS_INFO ("Subscribed to intensity image on: %s", intensity_img_sub.getTopic().c_str ());
  ROS_INFO ("Subscribed to depth image on: %s", depth_img_sub.getTopic().c_str ());
  ROS_INFO ("Subscribed to camera info on: %s", cam_info_sub.getTopic().c_str ());

  //cv::namedWindow("RGB");
  //cv::namedWindow("Depth");

  ros::spin();
}