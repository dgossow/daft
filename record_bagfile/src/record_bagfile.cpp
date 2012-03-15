#include "record_bagfile/record_bagfile.h"

RecordBagfile::RecordBagfile(ros::NodeHandle comm_nh, ros::NodeHandle param_nh) :
        comm_nh_( comm_nh ),
        param_nh_( param_nh ),
        rgbd_sync_( RgbdSyncPolicy(5), rgb_img_sub_, depth_img_sub_, cam_info_sub_),
        img_count_(1),
        subscribed_(false)
{
  std::string bagfile_name;
  if ( !param_nh_.getParam("bagfile", bagfile_name) )
  {
    std::cout << "Please specify a name for the output bagfile using bagfile:=<value>" << std::endl;
    ros::shutdown();
    return;
  }

  std::cout << "Writing a bagfile to "<< bagfile_name.c_str() << " " << std::endl;

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
  rgb_img_sub_.subscribe(comm_nh_, "rgb_image", 10);
  depth_img_sub_.subscribe(comm_nh_, "depth_image", 10);
  cam_info_sub_.subscribe(comm_nh_, "camera_info", 10);

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

void RecordBagfile::recordBagfileCB(const sensor_msgs::Image::ConstPtr rgb_img,
                                    const sensor_msgs::Image::ConstPtr depth_img,
                                    const sensor_msgs::CameraInfo::ConstPtr cam_info)
{
  std::cout << "Writing data set #"<< img_count_ << " " << std::endl;

  ros::Time current_time( img_count_ );

  bag_.write("rgb_img", current_time, rgb_img);
  bag_.write("depth_img", current_time, depth_img);
  bag_.write("cam_info", current_time, cam_info);

  img_count_++;
  unsubscribe();
}



