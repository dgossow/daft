/*
 * apply_blur_noise.h
 *
 *  Created on: Mar 16, 2012
 *      Author: praktikum
 */

#ifndef APPLY_BLUR_NOISE_H_
#define APPLY_BLUR_NOISE_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace rgbd_evaluator
{

class ApplyBlurNoise
{

public:
  ApplyBlurNoise( std::string file_path );
  virtual ~ApplyBlurNoise();

private:
  void noiseImage();
  void blurImage();
  void writeMaskPointFile( uint32_t imageNumber, uint32_t count );
  void writeHomographyToFile( cv::Matx33f homography, uint32_t count );
  void writeVectorToFile( std::vector<float> vec, std::string filename );
  void splitFileName( const std::string& str );
  void gaussianRandGen(cv::Mat& image, uint32_t sigma);
  std::string extractDigit( std::string& str );

  cv::Mat image_in_;

  std::string file_path_;
  std::string file_name_;
  std::string file_folder_;
  std::string file_created_folder_;
  std::string file_created_sub_folder_;

  static const uint32_t NUMBER_IMAGES_BLURRED = 8;
  static const uint32_t NUMBER_IMAGES_NOISED = 6;
  static const uint32_t NUMBER_MASK_POINTS = 4;

};

} /* namespace rgbd_evaluator */

#endif /* APPLY_BLUR_NOISE_H_ */
