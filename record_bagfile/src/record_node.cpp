
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

  RecordBagfile recordData(comm_nh, param_nh);

  Rate spin_rate(1);

  while (ros::ok())
  {
    if ( !recordData.isSubscribed() )
    {
      ROS_INFO( "Press enter to record an image or 'q'+enter to quit." );
      if ( getchar() == 'q' )
      {
        ros::shutdown();
      }
      else
      {
        recordData.subscribe();
      }
    }

    spinOnce();
  }

  std::cout << "Exiting..." << std::endl;
}

