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
#include <math.h>

#include <LinearMath/btQuaternion.h>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btVector3.h>

namespace rgbd_evaluator
{

RgbdEvaluatorPreprocessing::RgbdEvaluatorPreprocessing(std::string file_path)
{
  std::cout << "Reading bagfile from " << file_path.c_str() << std::endl;

  splitFileName(file_path);

  // create folder to store images and homography --> should be in /home/... otherwise root permissions neccessary
  std::string makeFolder;
  makeFolder.append("mkdir ");
  makeFolder.append(file_created_folder_);

  if( system(makeFolder.c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << makeFolder  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  bag_.open(file_path, rosbag::bagmode::Read);

}

RgbdEvaluatorPreprocessing::~RgbdEvaluatorPreprocessing()
{
  std::cout << "Stopping preprocessing..." << std::endl;

  cv::destroyAllWindows();

  bag_.close();
}

void RgbdEvaluatorPreprocessing::createTestFiles()
{
  uint32_t count = 0;
  bool got_cam_info = false;

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back("rgb_img");
  topics.push_back("center_transform");
  topics.push_back("cam_info");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  sensor_msgs::Image::ConstPtr p_current_img;
  geometry_msgs::TransformStamped::ConstPtr p_current_transform;
  sensor_msgs::CameraInfo::ConstPtr p_cam_info;

  image_store_.push_back( ImageData() );

  // Load all messages into our stereo dataset
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      //check if rgb_img message arrived
      if (p_rgb_img != NULL)
      {

        if ( image_store_.back().image )
        {
          std::cout << "There is an imfile_path_age already for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        std::string fileName;

        // convert integer to string
        std::stringstream ss;
        ss << count;

        fileName.append(file_created_folder_);
        fileName.append("/");
        fileName.append(file_folder_);
        fileName.append("_");

        fileName.append(ss.str());
        fileName.append(".ppm");

        std::cout << "Writing to "<< fileName << std::endl;

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);

        // store data in vectorImageData
        image_store_.back().image = ptr;

        // write data to file
        cv::imwrite(std::string(fileName),ptr->image );

      }

      /**********************************************************************************************************************/

      // load center transform
      geometry_msgs::TransformStamped::ConstPtr p_center_transform = m.instantiate<geometry_msgs::TransformStamped>();

      if ( p_center_transform != NULL )
      {
        //std::cout << "center_transform available" << std::endl;

        if ( image_store_.back().approx_transform )
        {
          std::cout << "There is already a transform for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        tf::StampedTransform transform;
        tf::transformStampedMsgToTF(*p_center_transform, transform);

        image_store_.back().approx_transform = boost::make_shared<btTransform>(transform);
      }

      /**********************************************************************************************************************/

      // load cam_info
      sensor_msgs::CameraInfo::ConstPtr p_cam_info = m.instantiate<sensor_msgs::CameraInfo>();

      if(( p_cam_info != NULL ) && ( got_cam_info == false ))
      {
        //std::cout << "camera_info available" << std::endl;

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
        //std::cout << "ImageData_"<< count << " complete! Press any key to continue\n\r" << std::endl;
        //getchar();
        count++;
      }
  }
}

void RgbdEvaluatorPreprocessing::calculateHomography()
{
  uint32_t count = 1;
  bool first_image = true;

  std::vector< ImageData >::iterator it;
  std::vector<cv::Point2f> feature_vector_original;
  std::vector<cv::KeyPoint> keypoint_vector_original;

  btTransform transform_original;
  btTransform transform_camx;
  btTransform transform_camx_to_original;

  cv::Mat image_original;
  cv::Mat image_grayscale;

  cv::Matx33f homography_complete;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = image_store_.begin(); it != image_store_.end()-1; it++)
  {

    // store first transforms and images
    if(first_image)
    {
      // get original image
      transform_original = *( it->approx_transform.get() );
      image_original = it->image.get()->image;
      first_image = false;

      // convert image to grayscale
      cv::cvtColor( image_original, image_grayscale, CV_RGB2GRAY );
      cv::goodFeaturesToTrack( image_grayscale, feature_vector_original, MAX_FEATURE_NUMBER, 0.01,
                               MIN_FEATURE_NEIGHBOUR_DIST, cv::noArray(), 3, true );

      cv::KeyPoint::convert( feature_vector_original, keypoint_vector_original );
      //cv::drawKeypoints( image_original, keypoint_vector_original, image_original );

      // store original corner points
//    cv::imshow("Original Image", image_original);
//    cv::waitKey(30);

      continue;
    }

    /**************************** calculate initial homography ***************************************************************/

    transform_camx = *(it->approx_transform.get());

    // calculate transform from camera position x to original position
    transform_camx_to_original = transform_original * transform_camx.inverse();

    cv::Matx33f homography_init = calculateInitialHomography( transform_camx_to_original, transform_camx );

    cv::Mat image_camx = it->image.get()->image;
    cv::Mat image_warped;

    // perspective warping
    cv::warpPerspective( image_camx, image_warped, cv::Mat(homography_init), cv::Size(image_original.cols,image_original.rows) );

    /**************************** calculate precise homography **************************************************************/

    std::vector<cv::Point2f> keypoints_camx;
    std::vector<cv::Point2f> keypoints_original;
    std::vector<cv::Point2f> feature_vector_camx;
    std::vector<cv::KeyPoint> keypoint_vector_camx;

    // convert image to grayscale
    cv::cvtColor( image_warped, image_grayscale, CV_RGB2GRAY );
    cv::goodFeaturesToTrack( image_grayscale, feature_vector_camx, MAX_FEATURE_NUMBER, 0.01, MIN_FEATURE_NEIGHBOUR_DIST );

    cv::KeyPoint::convert( feature_vector_camx, keypoint_vector_camx );
    cv::drawKeypoints( image_warped, keypoint_vector_camx, image_warped );

    // corner correspondences
    uint32_t correspondences[keypoint_vector_camx.size()];

    uint32_t i;

    for(i = 0; i < keypoint_vector_camx.size(); i++)
    {
      bool correspondence_found = false;

      // find correspondences
      uint32_t j=0;
      double_t min_distance = MAX_CORRESPONDENCES_DIST_THRES;
      correspondences[i] = 0;

      for(j = 0; j < keypoint_vector_original.size(); j++)
      {
          double_t tmp = calculateEuclidianDistance( keypoint_vector_original.at(j), keypoint_vector_camx.at(i) );

          if(min_distance >= tmp)
          {
              correspondences[i] = j;
              min_distance = tmp;
              correspondence_found = true;
          }
      }

      if(correspondence_found == true)
      {
          // store corresponding original point
          keypoints_camx.push_back( cv::Point2f( keypoint_vector_camx.at(i).pt.x,
                                                 keypoint_vector_camx.at(i).pt.y ) );

          keypoints_original.push_back( cv::Point2f( keypoint_vector_original.at(correspondences[i]).pt.x,
                                                     keypoint_vector_original.at(correspondences[i]).pt.y ) );
      }

    }

    printf("Found correspondences: %d\n\r", (uint32_t)keypoints_camx.size() );

    cv::Matx33f homography_precise;


    if(keypoints_camx.size() < MIN_CORRESPONDENCES || keypoints_original.size() < MIN_CORRESPONDENCES )
    {
      std::cout << "Not enough correspondences found! Exiting..." << std::endl;
      return;
    }

    homography_precise = cv::findHomography( keypoints_camx, keypoints_original , CV_LMEDS, 2 );

    homography_complete = homography_precise * homography_init;

    printMat( cv::Matx33f(homography_complete) );

    cv::Mat image_warped_precise;

    // perspective warping precise
    cv::warpPerspective( image_camx, image_warped_precise, cv::Mat(homography_complete), cv::Size(image_warped.cols,image_warped.rows) );


    //transform keypoints from warped image into precise warped image
    for ( uint32_t i=0; i < keypoints_camx.size(); i++ )
    {
      cv::Matx31f kp_xyw( keypoints_camx[i].x, keypoints_camx[i].y, 1);

      kp_xyw = homography_precise * kp_xyw;
      //corner_vector_src;
      keypoints_camx[i].x = kp_xyw.val[0] / kp_xyw.val[2];
      keypoints_camx[i].y = kp_xyw.val[1] / kp_xyw.val[2];
    }

    cv::Mat image_original_clone( image_original.clone() );

    // show error via lines
    for ( uint32_t i=0; i<keypoints_camx.size(); i++ )
    {
      cv::line( image_warped_precise, keypoints_camx[i], keypoints_original[i], cv::Scalar(0,0,255), 1 );
      cv::line( image_original_clone, keypoints_camx[i], keypoints_original[i], cv::Scalar(0,0,255), 1 );
    }

    /*************************************************************************************************************************/

    // show images
//    cv::imshow("Current Image", image_camx);
//    cv::waitKey(30);
//    cv::imshow("Warped Image", image_warped);
//    cv::waitKey(30);

    cv::imshow( "Precise warped Image", image_warped_precise );
    cv::waitKey(30);
    cv::imshow( "Original Image", image_original_clone );
    cv::waitKey(30);

    // store homography
    writeHomographyToFile( homography_complete, count++ );

    std::cout << "Press any key to continue" << std::endl;
    getchar();

  }

}

cv::Matx33f RgbdEvaluatorPreprocessing::calculateInitialHomography(btTransform transform_camx_to_original, btTransform transform_camx)
{
  // Translation
  tf::Point T_temp = transform_camx_to_original.getOrigin();
  cv::Matx31f T(T_temp.x(), T_temp.y(), T_temp.z());

  //Rotation
  btMatrix3x3 R_temp(transform_camx_to_original.getRotation());
  cv::Matx33f R( R_temp.getColumn(0).getX(), R_temp.getColumn(1).getX(), R_temp.getColumn(2).getX(),
                 R_temp.getColumn(0).getY(), R_temp.getColumn(1).getY(), R_temp.getColumn(2).getY(),
                 R_temp.getColumn(0).getZ(), R_temp.getColumn(1).getZ(), R_temp.getColumn(2).getZ());

  //N
  tf::Vector3 N_temp = transform_camx.getBasis() * btVector3(0,0,1);
  cv::Matx13f N(N_temp.x(), N_temp.y(), N_temp.z());

  //d
  T_temp = transform_camx.getOrigin();
  float_t d = ( N * (cv::Matx31f(T_temp.x(), T_temp.y(), T_temp.z())) ) (0);

  //Calculate init Homography
  cv::Matx33f homography_init = R + (1/d) * T * N;

  // + intrinsic-parameter-matrix
  homography_init = K_ * homography_init * K_.inv();

  return homography_init;
}


void RgbdEvaluatorPreprocessing::writeHomographyToFile(cv::Matx33f homography, uint32_t count)
{
  uint32_t i,j;
  std::fstream file;

  std::stringstream ss;
  ss << count;

  // create filepath
  std::string homographyName;
  homographyName.append(file_created_folder_);
  homographyName.append("/");
  homographyName.append("Homography_0_");
  homographyName.append(ss.str());
  homographyName.append(".dat");

  file.open(homographyName.c_str(), std::ios::out);

  for(i=0; i<3; i++)
  {
    for(j=0;j<3;j++)
    {
      file << homography(i,j) << "\t";
    }
    file << std::endl;
  }

  file.close();
}

void RgbdEvaluatorPreprocessing::printMat( cv::Matx33f M )
{
  std::cout << std::setprecision( 3 ) << std::right << std::fixed;
  for ( int row = 0; row < 3; ++ row )
  {
    for ( int col = 0; col < 3; ++ col )
    {
      std::cout << std::setw( 5 ) << (double)M( row, col ) << " ";
    }
    std::cout << std::endl;
  }
}

double_t RgbdEvaluatorPreprocessing::calculateEuclidianDistance(cv::KeyPoint corner_original, cv::KeyPoint corner_x)
{
  double_t edistance = sqrt(pow(corner_original.pt.x - corner_x.pt.x, 2) + pow(corner_original.pt.y - corner_x.pt.y, 2));
  return edistance;
}

void RgbdEvaluatorPreprocessing::splitFileName(const std::string& str)
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

} // end namespace


int main( int argc, char** argv )
{
  if(argc != 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << " <bagfileName>" << std::endl;
    return -1;
  }

  std::string fileName(argv[1]);

  rgbd_evaluator::RgbdEvaluatorPreprocessing fd(fileName);
  fd.createTestFiles();
  fd.calculateHomography();

  std::cout << "Exiting.." << std::endl;
  return 0;
}

