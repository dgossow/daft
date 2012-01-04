#include "record_bagfile/record_bagfile.h"


#include <tf/transform_datatypes.h>


RecordBagfile::RecordBagfile(ros::NodeHandle comm_nh, ros::NodeHandle param_nh) :
        comm_nh_( comm_nh ),
        param_nh_( param_nh ),
        //tf_listener_ ( comm_nh, ros::Duration(30) ),
        rgbd_sync_( RgbdSyncPolicy(5), rgb_img_sub_, depth_img_sub_, cam_info_sub_, point_cloud2_sub_ ),
        img_count_(1),
        subscribed_(false)
{
  std::string bagfile_name;
  if ( !param_nh_.getParam("bagfile", bagfile_name) )
  {
    ROS_ERROR( "Please specify a name for the output bagfile using bagfile:=<value>" );
    ros::shutdown();
    return;
  }

  ROS_INFO("Writing bagfile to '%s'", bagfile_name.c_str());

  bag_.open(bagfile_name, rosbag::bagmode::Write);

  rgbd_sync_.registerCallback( boost::bind( &RecordBagfile::recordBagfileCB, this, _1, _2 , _3, _4 ) );
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
  point_cloud2_sub_.subscribe(comm_nh_,"point_cloud", 10);

  ROS_INFO ("Subscribed to RGB image on: %s", rgb_img_sub_.getTopic().c_str ());
  ROS_INFO ("Subscribed to depth image on: %s", depth_img_sub_.getTopic().c_str ());
  ROS_INFO ("Subscribed to camera info on: %s", cam_info_sub_.getTopic().c_str ());
  ROS_INFO ("Subscribed to point cloud info on: %s", point_cloud2_sub_.getTopic().c_str ());

  subscribed_ = true;
}

void RecordBagfile::unsubscribe()
{
  rgb_img_sub_.unsubscribe();
  depth_img_sub_.unsubscribe();
  cam_info_sub_.unsubscribe();
  point_cloud2_sub_.unsubscribe();

  subscribed_ = false;
}

void RecordBagfile::recordBagfileCB(const sensor_msgs::Image::ConstPtr rgb_img,
                                    const sensor_msgs::Image::ConstPtr depth_img,
                                    const sensor_msgs::CameraInfo::ConstPtr cam_info,
                                    const sensor_msgs::PointCloud2::ConstPtr point_cloud)
{
  ROS_INFO( "Writing data set #%d", img_count_ );

  tf::StampedTransform center_transform;

  try {
    tf_listener_.waitForTransform(rgb_img->header.frame_id, "/marker_center", ros::Time(0), ros::Duration(2) );
    tf_listener_.lookupTransform( rgb_img->header.frame_id, "/marker_center", ros::Time(0), center_transform );
  }
  catch ( std::runtime_error err )
  {
          ROS_ERROR_THROTTLE( 1.0, "Error while looking up transform: %s", err.what() );
          return;
  }

  geometry_msgs::TransformStamped center_transform_msg;
  tf::transformStampedTFToMsg( center_transform, center_transform_msg );

  ros::Time current_time( img_count_ );

  bag_.write("rgb_img", current_time, rgb_img);
  bag_.write("depth_img", current_time, depth_img);
  bag_.write("cam_info", current_time, cam_info);
  bag_.write("point_cloud", current_time, point_cloud);
  bag_.write("center_transform", current_time, center_transform_msg);

  img_count_++;
  unsubscribe();
}



