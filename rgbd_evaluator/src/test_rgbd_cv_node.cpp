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

#include <daft2/daft.h>
#include <daft2/preprocessing.h>

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

  /*
  depth_image = orig_depth_image->image;

  // make intensity image smaller
  cv::Mat intensity_image_tmp = cv::Mat( orig_intensity_image->image, cv::Rect( 0,0, depth_image.cols*scale_fac, depth_image.rows*scale_fac ) );

  cv::resize( intensity_image_tmp, intensity_image, cvSize(depth_image.cols, depth_image.rows) );
  */

  //orig_depth_image->image

  cv::Mat1f depth_image_closed,depth_image_smoothed;
  cv::daft2::improveDepthMap<30>( depth_image, depth_image_closed, 0.2f );

  cv::Matx33d camera_matrix( ros_camera_info->P.data() );
  camera_matrix(1,2) /= 2;

  ROS_INFO_STREAM_ONCE( "f = " << camera_matrix(0,0) << " cx = " << camera_matrix(0,2) << " cy = " << camera_matrix(1,2) );

  cv::daft2::DAFT::DetectorParams p1,p2;
  std::vector<cv::KeyPoint3D> keypoints1,keypoints2;

  p1.base_scale_ = 0.005;
  p1.scale_levels_ = 1;
  p1.max_px_scale_ = 1000;

  p1.det_type_=p1.DET_DOB;
  p1.affine_=true;
  p1.max_search_algo_ = p1.MAX_WINDOW;

  p1.det_threshold_ = 0.04;

  p1.pf_type_ = p1.PF_PRINC_CURV_RATIO;
  p1.pf_threshold_ = 5;

  p2 = p1;
  //p1.det_type_=p1.DET_LAPLACE;
  p2.affine_ = false;
  p2.pf_type_ = p1.PF_NONE;

 cv::daft2::DAFT rgbd_features1(p1), rgbd_features2(p2);

#if 0
  // compare speeds
  for ( int i=0; i<10; i++ )
  {
    rgbd_features1.detect( intensity_image, depth_image, camera_matrix, keypoints1);
  }
  {
    boost::timer timer;
    timer.restart();
    for ( int i=0; i<10; i++ )
    {
      rgbd_features1.detect( intensity_image, depth_image, camera_matrix, keypoints1);
    }
    std::cout << "detect 1 execution time [ms]: " << timer.elapsed()*100 << std::endl;
  }
  for ( int i=0; i<10; i++ )
  {
    rgbd_features2.detect( intensity_image, depth_image, camera_matrix, keypoints1);
  }
  {
    boost::timer timer;
    timer.restart();
    for ( int i=0; i<10; i++ )
    {
      rgbd_features2.detect( intensity_image, depth_image, camera_matrix, keypoints2);
    }
    std::cout << "detect 2 execution time [ms]: " << timer.elapsed()*100 << std::endl;
  }

#else
  rgbd_features1.detect( intensity_image, depth_image_closed, camera_matrix, keypoints1);
  //rgbd_features2.detect( intensity_image, depth_image_closed, camera_matrix, keypoints2);
#endif

  //ROS_INFO_STREAM( keypoints1.size() << " / " << keypoints2.size() << " keypoints detected." );

#if 1
  // draw
  cv::Mat intensity_image1,intensity_image2;

  cv::drawKeypoints3D( intensity_image, keypoints2, intensity_image1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image1, keypoints1, intensity_image1, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  cv::drawKeypoints3D( intensity_image, keypoints1, intensity_image2, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image2, keypoints2, intensity_image2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  std::ostringstream s;
  s << p1.det_type_;
  cv::imshow( "KP1 (type "+s.str()+", Green) over KP2", intensity_image1 );

  static int f=0;
  s.str("");
  s << "/tmp/vid/";
  s.fill('0');
  s.width(5);
  s << f << ".jpg";
  cv::imwrite( s.str(), intensity_image1 );
  f++;

  s.str("");
  s << p2.det_type_;
  cv::imshow( "KP2 (type "+s.str()+", Red) over KP1", intensity_image2 );
#endif

#if 0
  cv::SURF surf;
  std::vector< cv::KeyPoint > surf_kp;
  cv::Mat mask;

  surf.hessianThreshold = 1250;

  surf( intensity_image, mask, surf_kp );

  cv::Mat surf_img;
  cv::drawKeypoints3D( intensity_image, keypoints1, surf_img, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints(surf_img, surf_kp, surf_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::imshow("SURF Keypoints", surf_img);
#endif
  cv::waitKey(500);
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
