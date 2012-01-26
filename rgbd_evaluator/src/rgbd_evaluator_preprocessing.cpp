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

#include <LinearMath/btQuaternion.h>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btVector3.h>

namespace rgbd_evaluator
{

RgbdEvaluatorPreprocessing::RgbdEvaluatorPreprocessing(std::string bagfile_name)
{
  std::cout << "Reading bagfile from " << bagfile_name.c_str() << std::endl;

  bagfile_name_ = bagfile_name;

  bag_.open(bagfile_name, rosbag::bagmode::Read);

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

        //std::cout << "rgb_img available" << std::endl;

        if ( image_store_.back().image )
        {
          std::cout << "There is an image already for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        char fileName[BUFF_SIZE];
        createFileName(fileName);

        sprintf(fileName, "%s_%d.ppm%c",fileName, count, '\0');

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
  std::vector<cv::Point> corner_vector_original;
  std::vector<cv::Point> corner_vector_camx;

  btTransform transform_original;
  btTransform transform_camx;
  btTransform transform_camx_to_original;

  cv::Mat image_original;
  cv::Mat harris_corners_original;
  cv::Mat harris_corners_camx_norm;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = image_store_.begin(); it != image_store_.end()-1; it++)
  {

    // store first transforms and images
    if(first_image)
    {
      // get original image
      transform_original = *(it->approx_transform.get());
      image_original = it->image.get()->image;
      first_image = false;

      // extract original harris corner points
      corner_vector_original = extractHarrisCornerPoints(image_original);

      cv::imshow("Original Image", image_original);
      cv::waitKey(30);

      continue;
    }

    /**************************** calculate initial homography ***************************************************************/

    transform_camx = *(it->approx_transform.get());

    // calculate transform from camera position x to original position
    transform_camx_to_original = transform_original * transform_camx.inverse();

    cv::Matx33f homography_init = calculateInitialHomography(transform_camx_to_original, transform_camx);

    cv::Mat image_camx = it->image.get()->image;
    cv::Mat image_warped;

    // perspective warping
    cv::warpPerspective(image_camx, image_warped, cv::Mat(homography_init), cv::Size(image_original.cols,image_original.rows));

    /**************************** calculate precise homography **************************************************************/

    std::vector<cv::Matx31f> corner_vector_camx_warped;

    corner_vector_camx = extractHarrisCornerPoints(image_camx);

    uint32_t i;
    for(i = 0; i < corner_vector_camx.size(); i++)
    {
      cv::Matx31f point_temp(corner_vector_camx.at(i).x, corner_vector_camx.at(i).y, 1.0);
      cv::Matx31f point_warped = cv::Matx33f(homography_init) * point_temp ;

      corner_vector_camx_warped.push_back( point_warped );

      cv::circle( image_warped, cv::Point( point_warped.val[0]/point_warped.val[2], point_warped.val[1]/point_warped.val[2] ), 5,  cv::Scalar(0), 2, 8, 0 );

    }

    // TODO: warp harris corner points from cam_x to cam_original; find corresponding points --> cv::findHomography




    /*************************************************************************************************************************/


    // show images
    cv::imshow("Current Image", image_camx);
    cv::waitKey(30);
    cv::imshow("Warped Image", image_warped);
    cv::waitKey(30);

    // store homography
    writeHomographyToFile(homography_init, count++);

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


std::vector<cv::Point> RgbdEvaluatorPreprocessing::extractHarrisCornerPoints(cv::Mat image)
{
  std::vector<cv::Point> keypoint_vector;

  cv::Mat image_grayscale;
  cv::Mat harris_corners;
  cv::Mat harris_corners_norm;

  double_t k = 0.04;
  uint32_t block_size = 5;
  uint32_t sobel_aperature = 3;
  int32_t threshold = 150;

  // convert image to grayscale
  cv::cvtColor(image, image_grayscale, CV_RGB2GRAY);
  cv::cornerHarris(image_grayscale, harris_corners, block_size, sobel_aperature, k);

  // normalizing
  cv::normalize( harris_corners, harris_corners_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );

  // extract harris corner points or draw a circle around corners
  for( int j = 0; j < harris_corners_norm.rows ; j++ )
  {
    for( int i = 0; i < harris_corners_norm.cols; i++ )
    {
       if( (int32_t) harris_corners_norm.at<float>(j,i) > threshold )
       {
         keypoint_vector.push_back(cv::Point(i,j));
         cv::circle( image, cv::Point( i, j ), 5,  cv::Scalar(0), 2, 8, 0 );
       }
    }
  }

  return keypoint_vector;
}

void RgbdEvaluatorPreprocessing::writeHomographyToFile(cv::Matx33f homography, uint32_t count)
{
  uint32_t i,j;
  std::fstream file;

  char fileName[BUFF_SIZE];
  char stdName[] = "Homography_0_to_";

  sprintf(fileName, "%s%d%c.dat",stdName, count, '\0');

  file.open(fileName, std::ios::out);

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

void RgbdEvaluatorPreprocessing::createFileName(char* fileName)
{
  uint32_t i = 0;

  while(bagfile_name_[i] != '.' && i < BUFF_SIZE)
  {
    fileName[i] = bagfile_name_.c_str()[i];
    i++;
  }
  fileName[i] = '\0';
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

