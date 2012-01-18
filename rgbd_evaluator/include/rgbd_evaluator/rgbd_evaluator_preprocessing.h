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
#include <geometry_msgs/TransformStamped.h>

#include <LinearMath/btTransform.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tf/tf.h>


namespace rgbd_evaluator
{

class RgbdEvaluatorPreprocessing
{
public:

  RgbdEvaluatorPreprocessing(std::string);
  virtual ~RgbdEvaluatorPreprocessing();

  void createTestFiles();
  void calculateHomography();

private:

  std::string bagfile_name_;
  rosbag::Bag bag_;
  cv_bridge::CvImagePtr tmp_image_;

  struct ImageData {
    boost::shared_ptr<cv_bridge::CvImage> image;
    boost::shared_ptr<btTransform> approx_transform;
    //...

    bool isComplete() {
      return image.get() && approx_transform.get();
    }
  };

  std::vector< ImageData > image_store_;
};

}
#endif /* RGBD_EVALUATOR_PREPROCESSING_H_ */
