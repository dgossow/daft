/*
 * extract_detector_file.h
 *
 *  Created on: Feb 20, 2012
 *      Author: praktikum
 */

#ifndef EXTRACT_DETECTOR_FILE_H_
#define EXTRACT_DETECTOR_FILE_H_

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rosbag/message_instance.h>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/foreach.hpp>

#include <rgbd_features_cv/daft.h>

namespace rgbd_evaluator
{

class ExtractDetectorFile
{
public:
  ExtractDetectorFile(std::string file_path);
  virtual ~ExtractDetectorFile();

private:
  void readBagFile();
  void extractKeypoints();
  void storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, uint32_t count);
  void splitFileName(const std::string& str);

  rosbag::Bag bag_;
  cv::DAFT daft_;
  cv::Matx33f K_;
  std::vector<cv::KeyPoint3D> keypoints_;

  std::string file_path_;
  std::string file_name_;
  std::string file_folder_;
  std::string file_created_folder_;

  struct ImageData
  {
    boost::shared_ptr<cv_bridge::CvImage> rgb_image;
    boost::shared_ptr<cv_bridge::CvImage> depth_image;

    bool isComplete()
    {
      return rgb_image.get() && depth_image.get();
    }
  };

  std::vector< ImageData > image_store_;
};

} /* namespace rgbd_evaluator */
#endif /* EXTRACT_DETECTOR_FILE_H_ */
