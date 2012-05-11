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

  cv::Matx33d camera_matrix( ros_camera_info->K.data() );

  ROS_INFO_STREAM_ONCE( "f = " << camera_matrix(0,0) << " cx = " << camera_matrix(0,2) << " cy = " << camera_matrix(1,2) );

  cv::daft2::DAFT::DetectorParams p1,p2;
  std::vector<cv::KeyPoint3D> keypoints1,keypoints2;

  p1.affine_ = true;
  p1.min_px_scale_ = 1.5;
  p1.det_threshold_ = 0.001;
  p1.pf_threshold_ = 7.5;
  p1.max_num_kp_ = 100;
  //p1.base_scale_ = 0.06;
  //p1.scale_levels_ = 1;

  p2 = p1;
  //p2.affine_ = false;
  //p2.pf_type_ = p2.PF_NONE;

  cv::daft2::DAFT rgbd_features1(p1), rgbd_features2(p2);

  cv::Mat1f desc;

  rgbd_features1( intensity_image, depth_image, camera_matrix, keypoints1, desc );
  rgbd_features2( intensity_image, depth_image, camera_matrix, keypoints2 );
  //

  //ROS_INFO_STREAM( keypoints1.size() << " / " << keypoints2.size() << " keypoints detected." );

#if 1
  // draw
  cv::Mat intensity_image1=intensity_image,intensity_image2=intensity_image;

  //cv::drawKeypoints3D( intensity_image, keypoints2, intensity_image1, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image1, keypoints1, intensity_image1, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  //cv::drawKeypoints3D( intensity_image, keypoints1, intensity_image2, cv::Scalar(0,255,0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  cv::drawKeypoints3D( intensity_image2, keypoints2, intensity_image2, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  std::ostringstream s;
  s << p1.det_type_;
  if ( keypoints1.size() != 0 )
  {
    cv::imshow( "keypoints1=green", intensity_image1 );
  }

  s.str("");
  s << p2.det_type_;
  if ( keypoints2.size() != 0 )
  {
    cv::imshow( "keypoints2=red", intensity_image2 );
  }

  /*
  static int f=0;
  s.str("");
  s << "/tmp/vid/";
  s.fill('0');
  s.width(5);
  s << f << ".jpg";
  cv::imwrite( s.str(), intensity_image1 );
  f++;
  */
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
  cv::waitKey(50);
}

using namespace cv;

int main( int argc, char** argv )
{
  /*
  const int PatchSize=20;
  float sum_weights = 0;
  const float center_uv = (float(PatchSize)-1.0f) * 0.5;
  for ( int v = 0; v<PatchSize; v++ )
  {
    for ( int u = 0; u<PatchSize; u++ )
    {
      // 0-centered u/v coords
      cv::Point2f uv( float(u)-center_uv, float(v)-center_uv );

      // normalized patch coords [-1,1]
      Point2f pt3d_uvw1 = uv * (2.0f/float(PatchSize));
      float dist_2 = pt3d_uvw1.x*pt3d_uvw1.x + pt3d_uvw1.y*pt3d_uvw1.y;

      const float weight = 1.0 - dist_2;
      if ( isnan(weight) || weight <= 0.0 )
      {
        continue;
      }

      sum_weights += weight;
    }
  }
  std::cout << sum_weights / double(PatchSize*PatchSize) << std::endl;
  */

  ros::init( argc, argv, "rgbd_evaluator_node" );

  ros::NodeHandle comm_nh(""); // for topics, services

  message_filters::Subscriber<sensor_msgs::Image> intensity_img_sub(comm_nh, "/camera/rgb/image_color", 1);
  message_filters::Subscriber<sensor_msgs::Image> depth_img_sub(comm_nh, "/camera/depth_registered/image", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub(comm_nh, "/camera/rgb/camera_info", 1);

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
#endif
