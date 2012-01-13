/*
 * rgbd_evaluator_preprocessing.h
 *
 *  Created on: Jan 13, 2012
 *      Author: praktikum
 */
#ifndef RGBD_EVALUATOR_PREPROCESSING_H_
#define RGBD_EVALUATOR_PREPROCESSING_H_

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>

#include <boost/foreach.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <sensor_msgs/Image.h>


namespace rgbd_evaluator
{

class RgbdEvaluatorPreprocessing
{
public:
  RgbdEvaluatorPreprocessing(std::string, ros::NodeHandle comm_nh, ros::NodeHandle param_nh);
  virtual ~RgbdEvaluatorPreprocessing();

private:
  std::string bagfile_name;
  rosbag::Bag bag_;
  ros::NodeHandle comm_nh_;
  ros::NodeHandle param_nh_;
};

}
#endif /* RGBD_EVALUATOR_PREPROCESSING_H_ */
