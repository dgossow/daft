#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <sensor_msgs/CameraInfo.h>

#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv.h>

using namespace ros;
using namespace sensor_msgs;


class RecordBagfile {

public:
    RecordBagfile(std::string bagfile_name, ros::NodeHandle comm_nh, ros::NodeHandle param_nh);

    ~RecordBagfile();

    void subscribe();
    void unsubscribe();

    bool isSubscribed() { return subscribed_; }

    void recordBagfileCB(const sensor_msgs::Image::ConstPtr rgb_img,
                         const sensor_msgs::Image::ConstPtr depth_img,
                         const sensor_msgs::CameraInfo::ConstPtr cam_info);

    static void mouseCb( int event, int x, int y, int flags, void* param );

private:

    bool recording_;
    int num_imgs_avg_;

    rosbag::Bag bag_;

    std::string win_name_;

    ros::NodeHandle comm_nh_;
    NodeHandle param_nh_;

    tf::TransformListener tf_listener_;

    message_filters::Subscriber<sensor_msgs::Image> rgb_img_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_img_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> RgbdSyncPolicy;

    message_filters::Synchronizer<RgbdSyncPolicy> rgbd_sync_;

    int img_count_;

    bool subscribed_;

    bool valid_;

    int cx_,cy_;
};
