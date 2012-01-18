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

  // Load all messages into our stereo dataset
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      //check if this message arrived
      if (p_rgb_img != NULL)
      {

        std::cout << "rgb_img available" << std::endl;

        if ( image_store_.back().image )
        {
          std::cout << "There is an image already for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        uint32_t i = 0;
        char fileName[500];

        while(bagfile_name_[i] != '.')
        {
          fileName[i] = bagfile_name_.c_str()[i];
          i++;
        }
        fileName[i] = '\0';


        sprintf(fileName, "%s_%d.ppm%c",fileName, count, '\0');

        std::cout << "Writing to "<< fileName << std::endl;

        // transform bag image to cvimage
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
        std::cout << "center_transform available" << std::endl;

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
  std::vector< ImageData >::iterator it;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = image_store_.begin(); it != image_store_.end()-1; it++)
  {
      std::cout << "Z: " << it->approx_transform.get()->getOrigin().z()  << std::endl;
  }

  // calculate homographys




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

