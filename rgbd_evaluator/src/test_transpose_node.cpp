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

#include <boost/timer.hpp>

#include <rgbd_features_cv/daft.h>
#include <rgbd_features_cv/preprocessing.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo > RgbdSyncPolicy;

cv_bridge::CvImageConstPtr orig_intensity_image;
cv_bridge::CvImageConstPtr orig_depth_image;
cv::Mat depth_image, intensity_image;

//#define TRANSPOSE_IMAGE

void rgbdImageCb(const sensor_msgs::Image::ConstPtr ros_intensity_image,
          const sensor_msgs::Image::ConstPtr ros_depth_image,
          const sensor_msgs::CameraInfo::ConstPtr ros_camera_info )
{
  ROS_INFO_STREAM_ONCE( "Image received." );

  orig_intensity_image = cv_bridge::toCvCopy( ros_intensity_image, "mono8" );
  orig_depth_image = cv_bridge::toCvCopy( ros_depth_image, sensor_msgs::image_encodings::TYPE_32FC1 );

  int scale_fac = orig_intensity_image->image.cols / orig_depth_image->image.cols;

  // Resize depth to have the same width as rgb
  cv::resize( orig_depth_image->image, depth_image, cvSize(0,0), scale_fac, scale_fac, cv::INTER_LINEAR );

  // Crop rgb so it has the same size as depth
  intensity_image = cv::Mat( orig_intensity_image->image, cv::Rect( 0,0, depth_image.cols, depth_image.rows ) );

  cv::Mat1f depth_image_closed;
  improveDepthMap<30>( depth_image, depth_image_closed, 0.2f );

  cv::Matx33d camera_matrix( ros_camera_info->P.data() );
  camera_matrix(1,2) /= 2;

  cv::Matx33d camera_matrix_t = camera_matrix;
  camera_matrix_t(0,2) = camera_matrix(1,2);
  camera_matrix_t(1,2) = camera_matrix(0,2);

  ROS_INFO_STREAM_ONCE( "f = " << camera_matrix(0,0) << " cx = " << camera_matrix(0,2) << " cy = " << camera_matrix(1,2) );

  cv::DAFT::DetectorParams p1,p2,p3;
  std::vector<cv::KeyPoint3D> keypoints,keypoints_t;

  p1.base_scale_ = 0.025;
  p1.scale_levels_ = 1;
  p1.max_px_scale_ = 1000;

  p1.det_type_=p1.DET_LAPLACE;
  p1.affine_=true;
  p1.max_search_algo_ = p1.MAX_FAST;

  p1.pf_type_ = p1.PF_NONE;

  cv::DAFT daft(p1);

  daft.detect( intensity_image, depth_image_closed, camera_matrix, keypoints);
  daft.detect( intensity_image.t(), depth_image_closed.t(), camera_matrix_t, keypoints_t);

  // draw
  cv::Mat intensity_image1,intensity_image2;

  cv::drawKeypoints3D( intensity_image.t(), keypoints_t, intensity_image1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image1.t(), keypoints, intensity_image1, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  cv::drawKeypoints3D( intensity_image, keypoints, intensity_image2, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image2.t(), keypoints_t, intensity_image2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  std::ostringstream s;
  s << p1.det_type_;
  cv::imshow( "KP (Red), KP_t (Green)", intensity_image1 );
  cv::imshow( "KP_t (Green), KP (Red)", intensity_image2.t() );
  cv::waitKey(100);
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

  while( ros::ok() )
  {
    ros::spinOnce();
  }
}
