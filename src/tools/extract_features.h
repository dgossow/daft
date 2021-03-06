/*
 * extract_detector_file.h
 *
 *  Created on: Feb 20, 2012
 *      Author: praktikum
 */

#ifndef EXTRACT_FEATURES_H_
#define EXTRACT_FEATURES_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/foreach.hpp>

#include <daft/daft.h>

#include "extract_impl.h"

namespace rgbd_evaluator
{

class ExtractDetectorFile
{
public:
  ExtractDetectorFile(std::string file_path,bool reset_files,bool verbose,bool small,int num_kp);
  virtual ~ExtractDetectorFile();

private:
  void readBagFile();

  void readDataFiles();

  bool fileExists( const std::string & fileName );

  bool readMatrix( const std::string & fileName, cv::Matx33f& K );

  bool readDepth( const std::string & fileName, cv::Mat1f& depth_img );

  void extractAllKeypoints();

  void extractKeypoints( GetKpFunc getKp, std::string name, float t );

  void storeKeypoints(
      std::vector<cv::KeyPoint3D> keypoints,
      cv::Mat1f& descriptors,
      std::string img_name,
      std::string extension,
      cv::Mat& rgb_img,
      cv::Mat& warped_img );

  std::vector<cv::KeyPoint3D> filterKpMask( std::vector<cv::KeyPoint3D> kp );

  void splitFileName(const std::string& str);

  void printMat( cv::Matx33f M );

  cv::Matx33f K_;
  cv::Mat1b mask_img_;

  std::string file_path_;
  std::string file_name_;
  std::string file_folder_;
  std::string file_created_folder_;
  std::string extra_folder_;
  std::string kp_folder_;

  int target_num_kp_;

  bool verbose_;
  bool small_;

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
