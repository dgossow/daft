/*
 * extract_detector_file.cpp
 *
 *  Created on: Feb 20, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/extract_detector_file.h"

#include <iostream>
#include <fstream>

#include <math.h>

#include <sensor_msgs/image_encodings.h>

namespace rgbd_evaluator
{

ExtractDetectorFile::ExtractDetectorFile(std::string file_path)
{
  std::cout << "Starting extract_detector_file..." << std::endl;

  splitFileName(file_path);

  // create folder to store detector file
  std::string makeFolder;
  makeFolder.append("mkdir ");
  makeFolder.append(file_created_folder_);

  if( system(makeFolder.c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << makeFolder  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  cv::DAFT::DetectorParams params;
  daft_ = cv::DAFT( params );

  bag_.open(file_path, rosbag::bagmode::Read);

  readBagFile();

  extractKeypoints();

}

ExtractDetectorFile::~ExtractDetectorFile()
{
  std::cout << "Stopping extract_detector_file..." << std::endl;

  bag_.close();
}

void ExtractDetectorFile::readBagFile()
{
  bool got_cam_info = false;

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back("rgb_img");
  topics.push_back("depth_img");
  topics.push_back("cam_info");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  image_store_.push_back( ImageData() );

  // Load all messages into our stereo dataset
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      //check if rgb_img message arrived
      if (p_rgb_img != NULL && p_rgb_img->encoding == "bgr8" )
      {

        if ( image_store_.back().rgb_image )
        {
          std::cout << "There is already an rgb image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);

        // store data in vectorImageData
        image_store_.back().rgb_image = ptr;

      }

      /**********************************************************************************************************************/

      // load depth image
      sensor_msgs::Image::ConstPtr p_depth_img = m.instantiate<sensor_msgs::Image>();

      //check if depth_img message arrived
      if (p_depth_img != NULL && p_depth_img->encoding == "32FC1" )
      {

        if ( image_store_.back().depth_image )
        {
          std::cout << "There is already an depth image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_depth_img);

        // store data in vectorImageData
        image_store_.back().depth_image = ptr;

      }

      /**********************************************************************************************************************/

      // load cam_info
      sensor_msgs::CameraInfo::ConstPtr p_cam_info = m.instantiate<sensor_msgs::CameraInfo>();

      if(( p_cam_info != NULL ) && ( got_cam_info == false ))
      {
        boost::array<double,9> cam_info = p_cam_info->K;

        K_ = cv::Matx33f(cam_info.at(0), cam_info.at(1), cam_info.at(2),
                         cam_info.at(3), cam_info.at(4), cam_info.at(5),
                         cam_info.at(6), cam_info.at(7), cam_info.at(8));

        got_cam_info = true;
      }

      /**********************************************************************************************************************/

      // if the current image data is complete, go to next one
      if ( image_store_.back().isComplete() )
      {
        image_store_.push_back( ImageData() );
      }
  }
}

void ExtractDetectorFile::extractKeypoints()
{
  uint32_t count = 0;

  std::vector< ImageData >::iterator it;

  // -1 because of initial push back
  for (it = image_store_.begin(); it != image_store_.end()-1; it++)
  {
    cv::Mat bag_rgb_img = it->rgb_image.get()->image;
    cv::Mat bag_depth_img = it->depth_image.get()->image;

    cv::Mat rgb_img;
    cv::Mat depth_img;

    int scale_fac = bag_rgb_img.cols / bag_depth_img.cols;

    // Resize depth to have the same width as rgb
    cv::resize( bag_depth_img, depth_img, cvSize(0,0), scale_fac, scale_fac, cv::INTER_LINEAR );

    // Crop rgb so it has the same size as depth
    rgb_img = cv::Mat( bag_rgb_img, cv::Rect( 0,0, depth_img.cols, depth_img.rows ) );

    cv::Mat img_out, greyscale_img;

    daft_.detect(rgb_img, depth_img, K_, keypoints_);

    cv::drawKeypoints3D(rgb_img, keypoints_, img_out, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    //cv::imshow("KEYPOINTS", img_out);
    //cv::waitKey(30);

    storeKeypoints(keypoints_, ++count);

    //std::cout << "Press any Key to continue!" << std::endl;
    //getchar();
  }

}

void ExtractDetectorFile::storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, uint32_t count)
{
  std::vector< cv::KeyPoint3D >::iterator it;
  double_t ax, bx, ay, by, a_length, b_length, alpha_a, alpha_b;
  double_t A, B, C;

  // create filepath
  std::stringstream ss;
  ss << count;

  std::string filePath;
  filePath.append(file_created_folder_);
  filePath.append("/");
  filePath.append("img");
  filePath.append(ss.str());
  filePath.append(".daftaf");

  // open file
  std::fstream file;
  file.open(filePath.c_str(), std::ios::out);

  // header
  file << "1.0" << std::endl;
  file << keypoints.size() << std::endl;

  for ( it = keypoints.begin(); it != keypoints.end(); it++ )
  {
    cv::Matx22f mat_affine;// = it->affine_mat;

    ax = mat_affine(0,0);
    bx = mat_affine(1,0);
    ay = mat_affine(0,1);
    by = mat_affine(1,1);

    alpha_a = atan2(ay,ax);
    alpha_b = atan2(by,bx);

    a_length = sqrt(pow(ax,2)+pow(ay,2));
    b_length = sqrt(pow(bx,2)+pow(by,2));

    ax = cos(alpha_a);
    bx = cos(alpha_b);
    ay = sin(alpha_a);
    by = sin(alpha_b);

//    std::cout << "ax: " << ax << "\tay: " << ay << "\tbx: " << bx << "\tby: " << by << "\t|a|: " << a_length << "\t|b|: " << b_length << std::endl;

    A = ( pow(ax,2) * pow(b_length,2) + pow(bx,2) * pow(a_length,2)) / (pow(a_length,2) * pow(b_length,2) );

    B = 2 * ( ( ax * ay * pow(b_length,2) + bx * by * pow(a_length,2)) ) / (pow(a_length,2) * pow(b_length,2) );

    C = ( pow(ay,2) * pow(b_length,2) + pow(by,2) * pow(a_length,2)) / (pow(a_length,2) * pow(b_length,2) );

//    std::cout << "x: " << it->pt.x << "\ty: " << it->pt.y << "\tA: " << A << "\tB: " << B << "\tC: " << C << std::endl;

    file << it->pt.x << "  " << it->pt.y << "  " << A << "  " << B << "  " << C << std::endl;

  }


  file.close();
}

void ExtractDetectorFile::splitFileName(const std::string& str)
{
  size_t found;
  std::cout << "Splitting: " << str << std::endl;
  found=str.find_last_of("/\\");

  file_path_ = str.substr(0,found);
  file_name_ = str.substr(found+1);

  found = file_name_.find_last_of(".");
  file_folder_ = file_name_.substr(0,found);

  file_created_folder_.append(file_path_);
  file_created_folder_.append("/");
  file_created_folder_.append(file_folder_);

  std::cout << " path: " << file_path_ << std::endl;
  std::cout << " file: " << file_name_ << std::endl;
  std::cout << " folder: " << file_folder_ << std::endl;
  std::cout << " created folder: " << file_created_folder_ << std::endl;
}

} /* namespace rgbd_evaluator */




int main( int argc, char** argv )
{
  if(argc != 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << " <bagfileName>" << std::endl;
    return -1;
  }

  std::string fileName(argv[1]);

  rgbd_evaluator::ExtractDetectorFile extract_detector_file(fileName);

  std::cout << "Exiting.." << std::endl;
  return 0;
}

