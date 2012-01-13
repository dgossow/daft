/*
 * rgbd_evaluator_preprocessing.cpp
 *
 *  Created on: Jan 13, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/rgbd_evaluator_preprocessing.h"

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

namespace rgbd_evaluator
{

RgbdEvaluatorPreprocessing::RgbdEvaluatorPreprocessing(std::string bagfile_name, ros::NodeHandle comm_nh, ros::NodeHandle param_nh) : comm_nh_( comm_nh ), param_nh_( param_nh )
{
  ROS_INFO("Reading bagfile from '%s'", bagfile_name.c_str());

  bag_.open(bagfile_name, rosbag::bagmode::Read);

  // Image topics to load
  std::string rgb_img = "rgb_img";
  std::vector<std::string> topics;
  topics.push_back(rgb_img);

  rosbag::View view(bag_, rosbag::TopicQuery(topics));
  // Load all messages into our stereo dataset

  uint32_t count = 0;

  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      if (p_rgb_img != NULL)
      {
        char fileName[500];
        sprintf(fileName, "%s%d%s%c",bagfile_name.c_str(), count, ".ppm",'\0');

        count++;

        ROS_INFO("Writing to %s", fileName);

        // write data to file
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);
        cv::imwrite(std::string(fileName),ptr->image );

      }
  }
  ros::shutdown();
}

RgbdEvaluatorPreprocessing::~RgbdEvaluatorPreprocessing()
{
  ROS_INFO("Stopping preprocessing...");
  bag_.close();
}

}

int main( int argc, char** argv )
{
  if(argc != 2)
  {
    ROS_INFO("Wrong usage, Enter: %s <bagfileName>\n\r",argv[0]);
    ros::shutdown();
    return -1;
  }

  std::string fileName(argv[1]);

  ros::init(argc, argv, "recordBagfile");

  ros::NodeHandle comm_nh(""); // for topics, services
  ros::NodeHandle param_nh("~");

  rgbd_evaluator::RgbdEvaluatorPreprocessing *fd = new rgbd_evaluator::RgbdEvaluatorPreprocessing(fileName, comm_nh, param_nh);

  while ( ros::ok() )
  {
        ros::spinOnce();
  }

  delete fd;
  ROS_INFO( "Exiting.." );
}

