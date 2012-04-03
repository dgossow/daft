
#include <stdio.h>
#include <ros/ros.h>

#include "record_bagfile/record_bagfile.h"

using namespace ros;
using namespace sensor_msgs;


int main(int argc, char **argv)
{
  init(argc, argv, "recordBagfile");

  ROS_INFO( "Starting bagfile recording..." );

  ros::NodeHandle comm_nh(""); // for topics, services
  ros::NodeHandle param_nh("~");

  if ( argc < 2 )
  {
    std::cout << "Usage: " << argv[0] << " <bagfile_name.bag>" << std::endl;
    return -1;
  }

  RecordBagfile recordData(argv[1],comm_nh, param_nh);
  recordData.subscribe();

  ros::spin();

  std::cout << "Exiting..." << std::endl;
}

