/*
* Copyright (C) 2011 David Gossow
*/


#include "rgbd_evaluator/rgbd_evaluator.h"

int main( int argc, char** argv )
{
  ros::init( argc, argv, "rgbd_evaluator_node" );

  ros::NodeHandle comm_nh(""); // for topics, services
  ros::NodeHandle param_nh("~");

  rgbd_evaluator::RgbdEvaluator * fd = new rgbd_evaluator::RgbdEvaluator(comm_nh, param_nh);

  ros::spin();

  delete fd;

  ROS_INFO( "Exiting.." );
}
