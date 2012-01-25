/*
 * Copyright (C) 2011 David Gossow
 */

//#include <stdlib.h>
//#include <math.h>

#include <opencv2/features2d/features2d.hpp>
#include <rgbd_features_cv/feature_detection.h>

namespace cv
{

void findMaxima( const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint >& kp )
{
  //#pragma omp parallel for
  //find maxima in sxs neighbourhood
  for ( int y = 3; y < img.rows-3; y++ )
  {
    for ( int x = 3; x < img.cols-3; ++x )
    {
      if ( img[y][x] < thresh || isnan( img[y][x] ) )
      {
        continue;
      }

      double s = scale_map[y][x] * base_scale;// * 0.25 - 1;

      if ( x-s < 0 || x+s >= img.cols || y-s < 0 || y+s > img.rows )
      {
        continue;
      }

      if (isnan( img[y-1][x-1] ) ||
          isnan( img[y-1][x  ] ) ||
          isnan( img[y-1][x+1] ) ||
          isnan( img[y  ][x-1] ) ||
          isnan( img[y  ][x  ] ) ||
          isnan( img[y  ][x+1] ) ||
          isnan( img[y+1][x-1] ) ||
          isnan( img[y+1][x  ] ) ||
          isnan( img[y+1][x+1] ))
      {
        continue;
      }

      int window = s / 2;
      if ( window < 1 ) window = 1;

      //static const int window = 1;

      bool isMax = true;
      for ( int v = 0; isMax && v <= window; v++ )
      {
        for ( int u = 0; isMax && u <= window; u++ )
        {
          if (u==0 && v==0)
          {
            continue;
          }
          if ( ( img[y+v][x+u] >= img[y][x] ) ||
              ( img[y+v][x-u] >= img[y][x] ) ||
              ( img[y-v][x+u] >= img[y][x] ) ||
              ( img[y-v][x-u] >= img[y][x] ) )
          {
            isMax=false;
          }
        }
      }

      if ( isMax )
      {
//#pragma omp critical
        kp.push_back( KeyPoint ( x, y, s, -1, img[y][x] ) );
      }
    }
  }
}

} 
