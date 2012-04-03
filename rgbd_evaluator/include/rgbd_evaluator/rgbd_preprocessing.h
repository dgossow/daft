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
#include <sensor_msgs/CameraInfo.h>

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

  RgbdEvaluatorPreprocessing(std::string, bool reverse_order, int start_img);
  virtual ~RgbdEvaluatorPreprocessing();

  void createTestFiles();
  void calculateHomography();

  std::vector<cv::Point2f> mouseKeypointsOrigin_;
  std::vector<cv::Point2f> mouseKeypointsImageX_;
  std::vector<cv::Point2f> mousePointsROI_;

  cv::Mat imageChooseROI_;
  cv::Mat keyPointImageOrigin_;
  cv::Mat keyPointImageCamX_;

  bool first_image_;
  bool finishedROI_;
  bool finishedKP_;

private:

  cv::Matx33f calculateInitialHomography(btTransform transform_camx_to_original, btTransform transform_original);
  cv::Matx33f calculateInitialHomography( cv::Mat& img1, cv::Mat& img2 );

  int32_t calculateNCC(cv::Mat image_original, cv::Mat image_cam_x, cv::KeyPoint keypoint, cv::Point2f& keypointNCC, int i);

  void printMat(cv::Matx33f M);

  void writeHomographyToFile(cv::Matx33f homography, uint32_t count);

  void writeIntrinsicMatToFile(cv::Matx33f K);

  void writeVectorToFile( std::vector<float> vec, std::string filename );

  void writeMaskPointsToFile( std::vector<cv::Point2f> maskPoints );

  void splitFileName (const std::string& str);

  void writeDepth( cv::Mat& depth_img_orig, std::string count_str );

  tf::StampedTransform calculateCoordinatesystem(cv::Mat& depth_img, std::vector<cv::Point2f> mouseKeypoints);

  static void imgMouseCallbackKP( int event, int x, int y, int flags, void* param );
  static void imgMouseCallbackROI( int event, int x, int y, int flags, void* param );

  std::string file_path_;
  std::string file_name_;
  std::string file_folder_;
  std::string file_created_folder_;

  bool reverse_order_;
  int start_img_;

  rosbag::Bag bag_;

  cv_bridge::CvImagePtr tmp_image_;

  cv::Matx33f K_;

  static const uint32_t BUFF_SIZE = 500;
  static const uint32_t MAX_CORRESPONDENCES_DIST_THRES = 10;
  static const uint32_t MIN_CORRESPONDENCES = 4;
  static const uint32_t MIN_FEATURE_NEIGHBOUR_DIST = 10;
  static const uint32_t MAX_FEATURE_NUMBER = 200;
  static const uint32_t SLIDING_WINDOW_SIZE = 40;
  static const uint32_t SEARCH_WINDOW_SIZE = SLIDING_WINDOW_SIZE+20;

  static const float_t  NCC_MAX_VAL = 0.98;

  struct ImageData
  {
    cv::Mat rgb_image;
    cv::Mat depth_image;

    bool isComplete()
    {
      return rgb_image.rows > 0 && depth_image.rows > 0;
    }
  };

  std::vector< ImageData > image_store_;
};

}
#endif /* RGBD_EVALUATOR_PREPROCESSING_H_ */
