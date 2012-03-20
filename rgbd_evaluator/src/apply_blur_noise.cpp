/*
 * apply_blur_noise.cpp
 *
 *  Created on: Mar 16, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/apply_blur_noise.h"

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

namespace rgbd_evaluator
{

ApplyBlurNoise::ApplyBlurNoise(std::string file_path)
{
  std::cout << "Started blur noise... " << std::endl;

  splitFileName(file_path);

  image_in_ = cv::imread(file_path);

  if ( image_in_.empty() )
  {
    std::cout << "Image not found!" << std::endl;
    return;
  }

  this->blurImage();
  this->noiseRGBImage();
  this->noiseDepthImage();
}

ApplyBlurNoise::~ApplyBlurNoise()
{
  cv::destroyAllWindows();
  std::cout << "Stopping blurring noise..." << std::endl;
}

void ApplyBlurNoise::noiseRGBImage()
{
    uint32_t i = 0;
    static const uint32_t sigmas[NUMBER_RGB_IMAGES_NOISED] = {10, 30, 50, 70, 90, 120};

    cv::Mat mIdentity = cv::Mat::eye(3, 3, CV_32S);

    std::vector<float> rotations;
    std::vector<float> scalings;
    std::vector<float> angles;

    // create noised_rgb folder
    std::string tmp = "mkdir ";
    file_created_sub_folder_.clear();
    file_created_sub_folder_.append(file_path_);
    file_created_sub_folder_.append("/");
    file_created_sub_folder_.append(file_folder_);
    file_created_sub_folder_.append("_");
    file_created_sub_folder_.append("noised_rgb");
    file_created_sub_folder_.append("/");

    tmp.append(file_created_sub_folder_);

    if( system( tmp.c_str() ) < 0) // -1 on error
    {
      std::cout << "Error when executing: " << file_created_sub_folder_  << std::endl;
      std::cout << "--> check user permissions"  << std::endl;
      return;
    }

    // read depth image and prepare copy to noise_rgb folder
    std::string imageNumber = extractDigit(file_folder_);
    std::string depthImagePath;
    depthImagePath.append(file_path_);
    depthImagePath.append("/depth");
    depthImagePath.append(imageNumber);
    depthImagePath.append(".pgm");

    cv::Mat depth_image = cv::imread(depthImagePath);

    // write MaskPoints to file
    writeMaskPointFile( NUMBER_RGB_IMAGES_NOISED, atoi(imageNumber.c_str()));

    for ( i = 0; i < NUMBER_RGB_IMAGES_NOISED; i++)
    {
      std::string imageFileName;
      std::string depthFileName;

      std::stringstream ss;
      ss << i+1;

      // create rgb image file name
      imageFileName.append(file_created_sub_folder_);
      imageFileName.append("img");
      imageFileName.append(ss.str());
      imageFileName.append(".ppm");

      // create depth image file name
      depthFileName.append(file_created_sub_folder_);
      depthFileName.append("depth");
      depthFileName.append(ss.str());
      depthFileName.append(".pgm");

      cv::Mat image_out = image_in_.clone();
      cv::Mat noise = image_in_.clone();

      // copy depth image
      cv::imwrite(depthFileName , depth_image);

      // noise rgb image
      cv::randn(noise,cv::Scalar::all(0),cv::Scalar::all(sigmas[i]));
      image_out = image_in_ + noise;

      // store noised rgb image
      cv::imwrite(imageFileName, image_out);

      // write homography
      if(i > 0)
      {
        writeHomographyToFile(mIdentity, i+1);
      }

      //fill vectors
      angles.push_back( 0.0 );
      rotations.push_back( 0.0 );
      scalings.push_back( 0.0 );
    }

    // write Scale, Rotation and Viewpoint angle to file (all 0)
    writeVectorToFile( rotations, "rotation" );
    writeVectorToFile( scalings, "scaling" );
    writeVectorToFile( angles, "viewpoint angle" );
}

void ApplyBlurNoise::noiseDepthImage()
{
    uint32_t i = 0;

    static const uint32_t sigmas[NUMBER_DEPTH_IMAGES_NOISED] = {10, 30, 50, 70, 90, 120};

    cv::Mat mIdentity = cv::Mat::eye(3, 3, CV_32S);

    std::vector<float> rotations;
    std::vector<float> scalings;
    std::vector<float> angles;

    // create noised_depth folder
    std::string tmp = "mkdir ";
    file_created_sub_folder_.clear();
    file_created_sub_folder_.append(file_path_);
    file_created_sub_folder_.append("/");
    file_created_sub_folder_.append(file_folder_);
    file_created_sub_folder_.append("_");
    file_created_sub_folder_.append("noised_depth");
    file_created_sub_folder_.append("/");

    tmp.append(file_created_sub_folder_);

    if( system( tmp.c_str() ) < 0) // -1 on error
    {
      std::cout << "Error when executing: " << file_created_sub_folder_  << std::endl;
      std::cout << "--> check user permissions"  << std::endl;
      return;
    }

    // read depth image and prepare copy to noised_depth folder
    std::string imageNumber = extractDigit(file_folder_);
    std::string depthImagePath;
    depthImagePath.append(file_path_);
    depthImagePath.append("/depth");
    depthImagePath.append(imageNumber);
    depthImagePath.append(".pgm");

    cv::Mat depth_img;
    readDepth(depth_img, depthImagePath);

    // write MaskPoints to file
    writeMaskPointFile( NUMBER_DEPTH_IMAGES_NOISED, atoi(imageNumber.c_str()));

    for ( i = 0; i < NUMBER_DEPTH_IMAGES_NOISED; i++)
    {

      std::string imageFileName;
      std::string depthFileName;

      std::stringstream ss;
      ss << i+1;

      // create image file name  std::fstream file;
      imageFileName.append(file_created_sub_folder_);
      imageFileName.append("img");
      imageFileName.append(ss.str());
      imageFileName.append(".ppm");

      // create depth  image file name
      depthFileName.append(file_created_sub_folder_);
      depthFileName.append("depth");
      depthFileName.append(ss.str());
      depthFileName.append(".pgm");

      cv::Mat image_out = depth_img.clone();
      cv::Mat noise = depth_img.clone();

      // copy rgb image
      cv::imwrite(imageFileName , image_in_);

      // noise depth image
      cv::randn(noise,cv::Scalar::all(0),cv::Scalar::all(sigmas[i]));
      image_out = depth_img - noise;

      cv::imshow("img_in", depth_img);
      cv::waitKey(30);
      cv::imshow("img_out", image_out);
      cv::waitKey();

      for(uint32_t p = 0; p < 10; p++)
      {
        std::cout << "img_in from ("<< i <<"," << p << "): " << depth_img.at<float>(i+200,p+200) << std::endl;
        std::cout << "image_out from ("<< i <<"," << p << "): " << image_out.at<float>(i+200,p+200) << std::endl;
      }

      // store noised depth image
      cv::imwrite(depthFileName, image_out);

      // write homography
      if(i > 0)
      {
        writeHomographyToFile(mIdentity, i+1);
      }

      //fill vectors
      angles.push_back( 0.0 );
      rotations.push_back( 0.0 );
      scalings.push_back( 0.0 );
    }

    // write Scale, Rotation and Viewpoint angle to file (all 0)
    writeVectorToFile( rotations, "rotation" );
    writeVectorToFile( scalings, "scaling" );
    writeVectorToFile( angles, "viewpoint angle" );
}

void ApplyBlurNoise::blurImage()
{
  uint32_t i = 0;
  static const uint32_t sigmas[NUMBER_IMAGES_BLURRED] = {1, 3, 5, 7, 9, 13, 17, 21 };

  cv::Mat mIdentity = cv::Mat::eye(3, 3, CV_32S);

  std::vector<float> rotations;
  std::vector<float> scalings;
  std::vector<float> angles;

  // create blurring folder
  std::string tmp = "mkdir ";
  file_created_sub_folder_.clear();
  file_created_sub_folder_.append(file_path_);
  file_created_sub_folder_.append("/");
  file_created_sub_folder_.append(file_folder_);
  file_created_sub_folder_.append("_");
  file_created_sub_folder_.append("blurred");
  file_created_sub_folder_.append("/");

  tmp.append(file_created_sub_folder_);

  if( system( tmp.c_str() ) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << file_created_sub_folder_  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  // read depth image and prepare copy to blurring folder
  std::string imageNumber = extractDigit(file_folder_);
  std::string depthImagePath;
  depthImagePath.append(file_path_);
  depthImagePath.append("/depth");
  depthImagePath.append(imageNumber);
  depthImagePath.append(".pgm");

  cv::Mat depth_image = cv::imread(depthImagePath);

  // write MaskPoints to file
  writeMaskPointFile( NUMBER_IMAGES_BLURRED ,atoi(imageNumber.c_str()));

  for ( i = 0; i < NUMBER_IMAGES_BLURRED; i++)
  {
    uint32_t sigma = sigmas[i];
    cv::Size ksize(3*sigma,3*sigma);

    std::string imageFileName;
    std::string depthFileName;

    std::stringstream ss;
    ss << i+1;

    // create image file name  std::fstream file;
    imageFileName.append(file_created_sub_folder_);
    imageFileName.append("img");
    imageFileName.append(ss.str());
    imageFileName.append(".ppm");

    // create depth  image file name
    depthFileName.append(file_created_sub_folder_);
    depthFileName.append("depth");
    depthFileName.append(ss.str());
    depthFileName.append(".pgm");

    cv::Mat image_out = image_in_.clone();

    // copy depth image
    cv::imwrite(depthFileName , depth_image);

    // blur image
    cv::GaussianBlur(image_in_, image_out, ksize, sigma, sigma);

    // store blurred image
    cv::imwrite(imageFileName, image_out);
    // write homography
    if(i > 0)
    {
      writeHomographyToFile(mIdentity, i+1);
    }

    //fill vectors
    angles.push_back( 0.0 );
    rotations.push_back( 0.0 );
    scalings.push_back( 0.0 );
  }

  // write Scale, Rotation and Viewpoint angle to file (all 0)
  writeVectorToFile( rotations, "rotation" );
  writeVectorToFile( scalings, "scaling" );
  writeVectorToFile( angles, "viewpoint angle" );
}

void ApplyBlurNoise::writeMaskPointFile(uint32_t imageNumber, uint32_t count)
{
  std::fstream file_oldMask, file_newMask;
  std::string maskPointPath;
  std::string newMaskPointPath;

  maskPointPath.append(file_path_);
  maskPointPath.append("/");
  maskPointPath.append("MaskPoints");

  newMaskPointPath.append(file_created_sub_folder_);
  newMaskPointPath.append("MaskPoints");

  std::cout << "Reading from: " << maskPointPath << std::endl;

  // open files
  file_oldMask.open(maskPointPath.c_str(), std::ios::in);
  file_newMask.open(newMaskPointPath.c_str(), std::ios::out);

  std::string tempLine[NUMBER_MASK_POINTS];
  uint32_t j = 0;

  for(uint32_t i = 0; i < count + NUMBER_MASK_POINTS; i++)
  {
    std::getline(file_oldMask, tempLine[j]);

    if(i >= count) j++;
  }

  for(uint32_t i = 0; i < imageNumber; i++)
  {
    file_newMask << tempLine[0] << std::endl;
    file_newMask << tempLine[1] << std::endl;
    file_newMask << tempLine[2] << std::endl;
    file_newMask << tempLine[3] << std::endl;
  }

}

void ApplyBlurNoise::writeHomographyToFile(cv::Matx33f homography, uint32_t count)
{
  uint32_t i,j;
  std::fstream file;

  std::stringstream ss;
  ss << count;

  // create filepath
  std::string homographyName;
  homographyName.append(file_created_sub_folder_);
  homographyName.append("/");
  homographyName.append("H1to");
  homographyName.append(ss.str());
  homographyName.append("p");

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

void ApplyBlurNoise::writeVectorToFile( std::vector<float> vec, std::string filename )
{
  uint32_t i;
  std::fstream file;

  // create filepath
  std::string vectorToFileName;
  vectorToFileName.append(file_created_sub_folder_);
  vectorToFileName.append(filename);

  file.open(vectorToFileName.c_str(), std::ios::out);

  for(i=0; i<vec.size(); i++)
  {
    file << vec[i] << "\t";
  }

  file.close();
}

void ApplyBlurNoise::splitFileName(const std::string& str)
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

std::string ApplyBlurNoise::extractDigit(std::string& str)
{
  std::string temp;

  for (uint32_t i=0; i < str.size(); i++)
  {
      if (isdigit(str[i]))
      {
        temp += str[i];
      }
  }

  return temp;
}

void ApplyBlurNoise::writeDepth( cv::Mat& depth_img_orig, std::string count_str )
{
  // write depth map
  // Convert float to 16-bit int
  cv::Mat1w depth_img;
  depth_img_orig.convertTo( depth_img, CV_16U, 1000.0, 0.0 );

  std::ofstream fs( (file_created_folder_ + "/" + "depth" + count_str + ".pgm").c_str() );

  fs << "P2" << std::endl;
  fs << depth_img.cols << " " << depth_img.rows << std::endl;
  fs << 65535 << std::endl;

  for ( int y=0; y<depth_img.rows; y++ )
  {
    for ( int x=0; x<depth_img.cols; x++ )
    {
      fs << depth_img[y][x] << " ";
    }
    fs << std::endl;
  }
}

void ApplyBlurNoise::readDepth(cv::Mat& depth_img, std::string ImagePath )
{
  uint32_t depth_rows = 0;
  uint32_t depth_cols = 0;

  std::string input_string;
  std::ifstream infile;
  infile.open(ImagePath.c_str(), std::ios::out);
  getline(infile,input_string); //Header1
  infile >> depth_cols;
  infile >> depth_rows;
  std::cout << "depth_rows: " << depth_rows << ", depth_cols: " << depth_cols << std::endl;
  getline(infile,input_string); //Header2

  cv::Mat depth_img_in;
  depth_img.create(depth_rows,depth_cols,CV_32F);

  for ( uint32_t y=0; y<depth_rows; y++ )
  {
    for ( uint32_t x=0; x<depth_cols; x++ )
    {
      infile >> depth_img.at<float>(y,x);
    }
  }
}


} /* namespace rgbd_evaluator */


int main( int argc, char** argv )
{
  if(argc < 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << " <ImagePath>" << std::endl;
    return -1;
  }

  std::string fileName = std::string(argv[1]);

  rgbd_evaluator::ApplyBlurNoise blurNoise(fileName);

  return 0;
}
