#include "record_bagfile/record_bagfile.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>


RecordBagfile::RecordBagfile(std::string bagfile_name, ros::NodeHandle comm_nh, ros::NodeHandle param_nh) :
        comm_nh_( comm_nh ),
        param_nh_( param_nh ),
        rgbd_sync_( RgbdSyncPolicy(5), rgb_img_sub_, depth_img_sub_, cam_info_sub_),
        img_count_(1),
        subscribed_(false)
{
  std::cout << "Writing a bagfile to "<< bagfile_name.c_str() << " " << std::endl;
  std::cout << "Press 'q' to exit." << std::endl;

  bag_.open(bagfile_name, rosbag::bagmode::Write);

  rgbd_sync_.registerCallback( boost::bind( &RecordBagfile::recordBagfileCB, this, _1, _2 , _3 ) );
}

RecordBagfile::~RecordBagfile()
{
  std::cout << "Finishing bagfile.." << std::endl;
  bag_.close();
}

void RecordBagfile::subscribe()
{
  rgb_img_sub_.subscribe(comm_nh_, "/camera/rgb/image_color", 10);
  depth_img_sub_.subscribe(comm_nh_, "/camera/depth_registered/image", 10);
  cam_info_sub_.subscribe(comm_nh_, "/camera/rgb/camera_info", 10);

  std::cout << "Subscribed to RGB image on: "<< rgb_img_sub_.getTopic().c_str() << " " << std::endl;
  std::cout << "Subscribed to depth image on: "<< depth_img_sub_.getTopic().c_str() << " " << std::endl;
  std::cout << "Subscribed to camera info on: "<< cam_info_sub_.getTopic().c_str() << " " << std::endl;

  subscribed_ = true;
}

void RecordBagfile::unsubscribe()
{
  rgb_img_sub_.unsubscribe();
  depth_img_sub_.unsubscribe();
  cam_info_sub_.unsubscribe();

  subscribed_ = false;
}

btVector3 getPt3D( int u, int v, float z, float f_inv, float cx, float cy )
{
  float zf = z*f_inv;
  btVector3 p;
  p[0] = zf * (u-cx);
  p[1] = zf * (v-cy);
  p[2] = z;
  return p;
}

bool getTransform( cv::Mat1f& depth_img,
    std::vector<cv::Point2f> img_pos, cv::Matx33f K,
    tf::StampedTransform& transform )
{
  float f_inv = 1.0 / K(0,0);
  float cx  = K(0,2);
  float cy  = K(1,2);

  std::vector<btVector3> CooPoint;
  btVector3 center;

  for(uint32_t i = 0; i < img_pos.size(); i++)
  {
    float num_zval = 0;
    float z_sum=0;
    for ( int y=-10;y<10;y++ )
    {
      for ( int x=-10;x<10;x++ )
      {
        float z = depth_img.at<float>( img_pos.at(i).y+y,img_pos.at(i).x+x );
        if ( !isnan(z) )
        {
          z_sum+=z;
          num_zval++;
        }
      }
    }

    if (num_zval == 0)
    {
      std::cout << "no depth value available!!!" << std::endl;
      return false;
    }

    float z = z_sum / num_zval;

    btVector3 CooPoint_tmp = getPt3D(
        img_pos.at(i).x,
        img_pos.at(i).y,
        z, f_inv, cx, cy );
    CooPoint.push_back( CooPoint_tmp );

    center += CooPoint_tmp;
  }

  center /= float(img_pos.size());

  btVector3 u = CooPoint[1] - CooPoint[0];
  btVector3 v = CooPoint[2] - CooPoint[0];
  btVector3 w = u.cross(v);
  btVector3 v1 = w.cross( u );

  btMatrix3x3 basis;
  basis[0] = u.normalize();
  basis[1] = v1.normalize();
  basis[2] = w.normalize();
  basis=basis.transpose();

  transform.setOrigin( center );
  transform.setBasis( basis );

  return true;

}

void RecordBagfile::recordBagfileCB(const sensor_msgs::Image::ConstPtr rgb_img_msg,
                                    const sensor_msgs::Image::ConstPtr depth_img_msg,
                                    const sensor_msgs::CameraInfo::ConstPtr cam_info_msg)
{
  cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(rgb_img_msg);

  cv_bridge::CvImagePtr orig_intensity_image = cv_bridge::toCvCopy( rgb_img_msg, sensor_msgs::image_encodings::BGR8 );
  cv_bridge::CvImagePtr orig_depth_image = cv_bridge::toCvCopy( depth_img_msg, sensor_msgs::image_encodings::TYPE_32FC1 );

  int scale_fac = orig_intensity_image->image.cols / orig_depth_image->image.cols;

  cv::Mat1f depth_img;
  cv::Mat3b bgr_img;

  // Resize depth to have the same width as rgb
  cv::resize( orig_depth_image->image, depth_img, cvSize(0,0), scale_fac, scale_fac, cv::INTER_NEAREST );

  // Crop rgb so it has the same size as depth
  bgr_img = cv::Mat( orig_intensity_image->image, cv::Rect( 0,0, depth_img.cols, depth_img.rows ) );

  for(int y = 0; y < depth_img.rows; y++)
  {
    for(int x = 0; x < depth_img.cols; x++)
    {
      if ( isnan( depth_img[y][x] ) )
      {
        bgr_img(y,x) = cv::Vec3b( bgr_img(y,x)[1]*(((y+x)/2)%2), 0, 0 );
      }
    }
  }

  boost::array<double,9> cam_info = cam_info_msg->K;
  cv::Matx33f K = cv::Matx33f(cam_info.at(0), cam_info.at(1), cam_info.at(2),
                   cam_info.at(3), cam_info.at(4), cam_info.at(5),
                   cam_info.at(6), cam_info.at(7), cam_info.at(8));

  const int w = 40;

  cv::line( bgr_img,
      cv::Point2f(depth_img.cols/2-w,depth_img.rows/2),
      cv::Point2f(depth_img.cols/2+w,depth_img.rows/2),
      cv::Scalar(0,255,0),
      3 );
  cv::line( bgr_img,
      cv::Point2f(depth_img.cols/2,depth_img.rows/2-w),
      cv::Point2f(depth_img.cols/2,depth_img.rows/2+w),
      cv::Scalar(0,255,0),
      3 );

  std::vector<cv::Point2f> img_pos;
  img_pos.push_back( cv::Point2f(depth_img.cols/2+w,depth_img.rows/2-w) );
  img_pos.push_back( cv::Point2f(depth_img.cols/2-w,depth_img.rows/2-w) );
  img_pos.push_back( cv::Point2f(depth_img.cols/2-w,depth_img.rows/2+w) );
  img_pos.push_back( cv::Point2f(depth_img.cols/2+w,depth_img.rows/2+w) );
  tf::StampedTransform transform;
  bool transform_valid = getTransform( depth_img, img_pos ,K, transform );

  btVector3 zvec = transform.getBasis() * btVector3(0,0,1);
  //btVector3 xvec = transform.getBasis() * btVector3(1,0,0);
  float dist = transform.getOrigin().length();

  if ( transform_valid )
  {
    std::stringstream s;
    s.precision( 3 );
    s << "dist = " << dist << " m";
    cv::putText( bgr_img, s.str( ), cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0) );

    //s.str("");
    //s << "rot  = " << xvec.angle( btVector3(1,0,0) ) / M_PI*180.0;
    //cv::putText( bgr_img, s.str( ), cv::Point(210,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0) );

    s.str("");
    s << "angle= " << zvec.angle( btVector3(0,0,-1) ) / M_PI*180.0 << " deg";
    cv::putText( bgr_img, s.str( ), cv::Point(310,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0) );
  }
  else
  {
    cv::putText( bgr_img, "Cannot determine transform.", cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255) );
  }

  cv::imshow( "image", bgr_img );

  int kb_input = cv::waitKey(100);
  //std::cout << kb_input << std::endl;

  if ( kb_input == 1048689 )
  {
    ros::shutdown();
  }
  else if ( kb_input >= 0 ) {
    std::cout << "Writing data set #"<< img_count_ << " " << std::endl;

    ros::Time current_time( img_count_ );

    bag_.write("rgb_img", current_time, rgb_img_msg);
    bag_.write("depth_img", current_time, depth_img_msg);
    bag_.write("cam_info", current_time, cam_info_msg);

    img_count_++;
  }
}



