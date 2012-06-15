/*
 * test1.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: gossow
 */

#include <opencv2/opencv.hpp>
#include <cmath>

#include "daft.h"
#include "filter_kernels.h"
#include "depth_filter.h"
#include "feature_detection.h"
#include "descriptor.h"
#include "preprocessing.h"
#include "gauss3d.h"

using namespace cv;
using namespace cv::daft2;
using namespace std;

#define DBG_OUT( SEQ ) std::cout << SEQ << std::endl



void makeSinoid( double lambda, cv::Mat1b sin_img )
{
  double ci = ((double)sin_img.rows-1) * 0.5;
  double cj = ((double)sin_img.cols-1) * 0.5;

  double t = M_PI*2.0 / lambda;

  for ( int i=0; i<sin_img.rows; i++ )
  {
    double i2 = (double)i - ci;
    for ( int j=0; j<sin_img.cols; j++ )
    {
      double j2 = (double)j - cj;
      sin_img( i, j ) = 255.0 * ( (cos(j2*t)+cos(i2*t)) / 4.0 + 0.5 );
    }
  }
}

void makeSinglePx( cv::Mat1b img, int cj, int ci )
{
  for ( int i=0; i<img.rows; i++ )
  {
    for ( int j=0; j<img.cols; j++ )
    {
      img( i, j ) = i==ci&&j==cj ? 255 : 0;
    }
  }
}

int main(int argc, char** argv)
{
  const int cols = 256;
  const int rows = 256;

  int ci = rows / 2;
  int cj = cols / 2;

  // camera matrix
  cv::Matx33f K = cv::Matx33f::eye();
  K(0,2) = (double)cols * 0.5;
  K(1,2) = (double)rows * 0.5;

  cv::Mat1f depth_map( rows, cols, 1.0f );

  // wavelength
  int lambda = 64;

  cv::Mat1b img( rows, cols );
  makeSinoid( lambda, img );
  //makeSinglePx( img, cj, ci );
  cv::imshow( "img", img );

  namedWindow( "smoothed_img" );
  createTrackbar("lambda", "smoothed_img", &lambda, cols );

  int old_lambda = 0;

  while( true )
  {
    if ( lambda != old_lambda )
    {
      old_lambda = lambda;

      cv::Mat1f smoothed_img;

      gauss3d<float,float,inter::linear<float>,inter::linear<float> >( K, depth_map, img, lambda/4.0, smoothed_img );
      imshowNorm( "smoothed_img", smoothed_img, -1 );
      imshowDxDy( "smoothed_img", smoothed_img, -1 );

      /*
      cv::Mat1f smoothed_img0,smoothed_img1;
      gauss3d<float,float,inter::linear<float>,inter::zero<float> >( K, depth_map, img, lambda/4.0, smoothed_img0 );
      imshowNorm( "smoothed_img0", smoothed_img0, -1 );
      gauss3d<float,float,inter::linear<float>,inter::one<float> >( K, depth_map, img, lambda/4.0, smoothed_img1 );
      imshowNorm( "smoothed_img1", smoothed_img1, -1 );
      */

      cv::Mat1f smoothed_img2;
      gauss3d<float,float,inter::linear<float>,inter::linear<float> >( K, depth_map, img, lambda/2.0, smoothed_img2 );
      imshowNorm( "smoothed_img2", smoothed_img2 );

      cv::Mat1f laplace = smoothed_img - smoothed_img2;
      imshowNorm( "laplace", laplace*0.5+0.5 );

      std::cout << "lambda = " << lambda << " laplace(center)=" << laplace(ci,cj) << std::endl;
    }
    cv::waitKey(100);
  }

  return 0;
}

