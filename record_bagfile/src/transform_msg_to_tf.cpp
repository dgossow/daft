#include <ros/ros.h>

#include <tf/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

tf::TransformBroadcaster* tf_broadcaster;

void transformCb( geometry_msgs::TransformStampedConstPtr center_transform_msg)
{
  tf::StampedTransform center_transform;
  tf::transformStampedMsgToTF( *center_transform_msg, center_transform );
  tf_broadcaster->sendTransform( center_transform );
}


int main( int argc, char** argv )
{
  ros::init( argc, argv, "transform_msg_to_tf" );
  ros::NodeHandle comm_nh("");

  tf_broadcaster = new tf::TransformBroadcaster();

  ros::Subscriber trans_sub = comm_nh.subscribe( "/center_transform", 10, &transformCb );

  ros::spin();
}
