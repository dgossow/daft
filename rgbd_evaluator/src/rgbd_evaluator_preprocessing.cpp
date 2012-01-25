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

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back("rgb_img");
  topics.push_back("center_transform");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  // create gui windows
  cv::namedWindow("Source Image", CV_WINDOW_AUTOSIZE);
  cv::waitKey(30);
  cv::namedWindow("Destination Image", CV_WINDOW_AUTOSIZE);
  cv::waitKey(30);

  sensor_msgs::Image::ConstPtr p_current_img;
  geometry_msgs::TransformStamped::ConstPtr p_current_transform;

  image_store_.push_back( ImageData() );

  // Load all messages into our stereo datas#include <math.h>et
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      //check if this message arrived
      if (p_rgb_img != NULL)
      {

        //std::cout << "rgb_img available" << std::endl;

        if ( image_store_.back().image )
        {
          std::cout << "There is an image already for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        uint32_t i = 0;

        char fileName[BUFF_SIZE];

        while(bagfile_name_[i] != '.' && i < BUFF_SIZE)
        {
          fileName[i] = bagfile_name_.c_str()[i];
          i++;
        }
        fileName[i] = '\0';


        sprintf(fileName, "%s_%d.ppm%c",fileName, count, '\0');

        std::cout << "Writing to "<< fileName << std::endl;

        // transform bag image to cvimage#include <math.h>
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);

        // store data in vectorImageData
        image_store_.back().image = ptr;

        // write data to file
        cv::imwrite(std::string(fileName),ptr->image );

        // show images
        cv::imshow("Source Image", image_store_.front().image->image);
        cv::waitKey(30);

        if(count >= 1)
        {
          cv::imshow("Destination Image", image_store_.back().image->image);
          cv::waitKey(30);
        }

      }

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

      // if the current image data is complete, go to next one
      if ( image_store_.back().isComplete() )
      {
        image_store_.push_back( ImageData() );
        std::cout << "ImageData_"<< count << " complete! Press any key to continue\n\r" << std::endl;
        count++;
        getchar();
      }
  }
}

void RgbdEvaluatorPreprocessing::calculateHomography()
{
  uint32_t count = 1;
  std::vector< ImageData >::iterator it;
  btTransform transformOrigin;
  btTransform transformCamX;
  cv::Mat imageOrigin;

  btTransform transform_cam_x_to_origin;

  bool first = true;

  std::cout << "Start transforming " << count << "..." << std::endl;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = image_store_.begin(); it != image_store_.end()-1; it++)
  {

    // store first transforms and images
    if(first)
    {
      transformOrigin = *(it->approx_transform.get());
      imageOrigin = it->image.get()->image;
      first = false;
      continue;
    }

    transformCamX = *(it->approx_transform.get());

    transform_cam_x_to_origin = transformCamX.inverse() * transformOrigin;

    cv::Matx33f homography_init = calculateInitialHomography(transform_cam_x_to_origin, transformOrigin);

    std::cout << "Warping image("<<imageOrigin.cols<<", "<<imageOrigin.rows<< ")"<< std::endl;

    cv::Mat tempMat = it->image.get()->image;

    cv::warpPerspective(imageOrigin, tempMat, cv::Mat(homography_init), cv::Size(imageOrigin.cols,imageOrigin.rows));//, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    cv::imshow("Source Image", imageOrigin);
    cv::waitKey(30);
    cv::imshow("Destination Image", tempMat);
    cv::waitKey(30);

//    std::cout << "Homography_0_to_"<< count << std::endl;
//
//    // temporary
//    int i,j;
//    for(i=0; i<3; i++)
//    {
//      for(j=0;j<3;j++)
//      {
//
//        std::cout << homography_init(i,j) << "  ";
//      }
//      std::cout << std::endl;
//    }

    writeHomographyToFile(homography_init, count);

    count++;
    std::cout << "Press any key to continue\n\r" << std::endl;
    getchar();

  }

}

cv::Matx33f RgbdEvaluatorPreprocessing::calculateInitialHomography(btTransform trans, btTransform transOrigin)
{
  float_t d;
  tf::Point p_temp = trans.getOrigin();

  // Translation
  cv::Matx31f T(p_temp.x(), p_temp.y(), p_temp.z());

  btMatrix3x3 m_temp(trans.getRotation());

  //Rotation
  cv::Matx33f R(m_temp.getRow(0).getX(), m_temp.getRow(0).getY(), m_temp.getRow(0).getZ(),
                           m_temp.getRow(1).getX(), m_temp.getRow(1).getY(), m_temp.getRow(1).getZ(),
                           m_temp.getRow(2).getX(), m_temp.getRow(2).getY(), m_temp.getRow(2).getZ());
  //d
  d = transOrigin.getOrigin().getZ();

  p_temp = transOrigin.getOrigin();
  //N
  cv::Matx13f N(p_temp.x(), p_temp.y(), p_temp.z());
  //Calculate init Homography
  cv::Matx33f homography_init = R + (1/d) * T * N;

  return homography_init;
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
}

