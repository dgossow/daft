/*
 * Copyright (C) 2011 David Gossow
 */

//#include <stdlib.h>
//#include <math.h>

#include <opencv2/features2d/features2d.hpp>

namespace cv
{

void findMaxima( const cv::Mat1d &filtered_image,
                 const cv::Mat1d &scale_map,
                 double base_scale,
                 std::vector< KeyPoint >& kp,
                 double threshold = 0.1 )
{
  //#pragma omp parallel for
  //find maxima in sxs neighbourhood
  for ( int y = 3; y < filtered_image.rows-3; y++ )
  {
    for ( int x = 3; x < filtered_image.cols-3; ++x )
    {
      if ( filtered_image[y][x] < threshold || isnan( filtered_image[y][x] ) )
      {
        continue;
      }

      double s = scale_map[y][x] * base_scale;// * 0.25 - 1;

      if ( x-s < 0 || x+s >= filtered_image.cols || y-s < 0 || y+s > filtered_image.rows )
      {
        continue;
      }

      if (isnan( filtered_image[y-1][x-1] ) ||
          isnan( filtered_image[y-1][x  ] ) ||
          isnan( filtered_image[y-1][x+1] ) ||
          isnan( filtered_image[y  ][x-1] ) ||
          isnan( filtered_image[y  ][x  ] ) ||
          isnan( filtered_image[y  ][x+1] ) ||
          isnan( filtered_image[y+1][x-1] ) ||
          isnan( filtered_image[y+1][x  ] ) ||
          isnan( filtered_image[y+1][x+1] ))
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
          if ( ( filtered_image[y+v][x+u] >= filtered_image[y][x] ) ||
              ( filtered_image[y+v][x-u] >= filtered_image[y][x] ) ||
              ( filtered_image[y-v][x+u] >= filtered_image[y][x] ) ||
              ( filtered_image[y-v][x-u] >= filtered_image[y][x] ) )
          {
            isMax=false;
          }
        }
      }

      if ( isMax )
      {
//#pragma omp critical
        kp.push_back( KeyPoint ( x, y, s, 0.0, filtered_image[y][x] ) );
      }
    }
  }
}

} 
