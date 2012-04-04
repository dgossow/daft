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

#include <daft2/daft.h>
#include <daft/daft.h>

namespace rgbd_evaluator
{

class ExtractDetectorFile
{
public:
  ExtractDetectorFile(std::string file_path,bool verbose,int num_kp);
  virtual ~ExtractDetectorFile();

private:
  void readBagFile();

  void readDataFiles();

  bool fileExists( const std::string & fileName );

  bool readMatrix( const std::string & fileName, cv::Matx33f& K );

  bool readDepth( const std::string & fileName, cv::Mat1f& depth_img );

  void extractAllKeypoints();

  typedef boost::function< std::vector<cv::KeyPoint3D> ( const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t ) > GetKpFunc;

  void extractKeypoints( GetKpFunc getKp, std::string name );

  void storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, std::string img_name, std::string extension, cv::Mat& rgb_img );

  std::vector<cv::KeyPoint3D> filterKpMask( std::vector<cv::KeyPoint3D> kp );

  void splitFileName(const std::string& str);

  void printMat( cv::Matx33f M );

  cv::Matx33f K_;
  cv::Mat maskImage_;

  std::string file_path_;
  std::string file_name_;
  std::string file_folder_;
  std::string file_created_folder_;
  std::string extra_folder_;

  int target_num_kp_;

  bool verbose_;

  struct ImageData
  {
    cv::Mat rgb_image;
    cv::Mat1f depth_image;
    cv::Matx33f hom;
  };

  std::vector< ImageData > image_store_;
};

} /* namespace rgbd_evaluator */
#endif /* EXTRACT_DETECTOR_FILE_H_ */
