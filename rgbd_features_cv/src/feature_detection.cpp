/*
 * Copyright (C) 2011 David Gossow
 */

#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>

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
      /*
      if ( img[y][x] < thresh || isnan( img[y][x] ) )
      {
        continue;
      }
*/
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

      int window = (s+0.5) / 2.0;
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

// Helper struct + comparison
struct MaxProp
{
  // function value
  float max_value;

  // index in original image
  unsigned int idx;

  // index inside (last) 2x2 block. Layout:
  // 0 1
  // 2 3
  unsigned int block_idx;

  // still needs check for local max
  bool need_check;

  friend bool operator< ( const MaxProp& lhs, const MaxProp& rhs )
  {
    return lhs.max_value < rhs.max_value;
  }
};

void findMaximaMipMap( const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint >& kp )
{
  if ( !img.isContinuous() )
  {
    return;
  }

  double max_dim = std::max( img.rows, img.cols );

  unsigned w = img.cols;
  unsigned h = img.rows;
  unsigned map_size = img.rows*img.cols;

  std::vector< MaxProp > last_map;

  //std::ofstream f;
  //f.open( "/tmp/wt.csv" );

  // compute max levels
  for ( int current_scale = 2; current_scale<max_dim; current_scale*=2 )
  {
    const unsigned w2 = w;
    w /= 2;
    h /= 2;
    map_size = w*h;

    if ( w<3 || h<3 ) break;

    std::vector< MaxProp > curr_map( map_size );

    // fixed value for rounding to next scale level
    // the factor is computed to minimize the error in
    // search area: sqrt(5/2) * 1.5
    float s_thresh = current_scale * 1.41 * 1.5;

    //std::cout << "Mipmap level " << current_scale << " thresh " << s_thresh << " size " << w << " x " << h << std::endl;

    const int diff[4][5] = {
        {  -w2-1,    -w2,  -w2+1,   -1,   w2-1 },
        {    -w2,  -w2+1,  -w2+2,    2,   w2+2 },
        {     -1,   w2-1, 2*w2-1, 2*w2, 2*w2+1 },
        {   2*w2, 2*w2+1, 2*w2+2, w2+2,      2 } };

    unsigned remaining_checks = 0;

    // compute max values in 2x2 blocks of higher level
    for ( unsigned y = 1; y < h-1; y++ )
    {
      for ( unsigned x = 1; x < w-1; ++x )
      {
        unsigned x2 = 2*x;
        unsigned y2 = 2*y;

        unsigned i = x + y*w;
        unsigned i2 = x2 + y2*w2;

        if ( current_scale == 2 )
        {
          double* last_map = reinterpret_cast<double*>(img.data);
          unsigned int max_idx12,max_idx34;
          unsigned int max_blockidx12,max_blockidx34;
          float max_val12,max_val34,max_val;

          if ( last_map[i2] > last_map[i2+1] )
          {
            max_val12 = last_map[i2];
            max_idx12 = i2;
            max_blockidx12 = 0;
          }
          else
          {
            max_val12 = last_map[i2+1];
            max_idx12 = i2+1;
            max_blockidx12 = 1;
          }
          if ( last_map[i2+w2] > last_map[i2+w2+1] )
          {
            max_val34 = last_map[i2+w2];
            max_idx34 = i2+w2;
            max_blockidx34 = 2;
          }
          else
          {
            max_val34 = last_map[i2+w2+1];
            max_idx34 = i2+1+w2;
            max_blockidx34 = 3;
          }
          if ( max_val12 > max_val34 )
          {
            MaxProp& p = curr_map[i];
            p.max_value = max_val12;
            p.idx = max_idx12;
            p.block_idx = max_blockidx12;
            p.need_check = max_val12 > thresh;
          }
          else
          {
            MaxProp& p = curr_map[i];
            p.max_value = max_val34;
            p.idx = max_idx34;
            p.block_idx = max_blockidx34;
            p.need_check = max_val34 > thresh;
          }
        }
        else
        {
          MaxProp max_12 = std::max( last_map[i2], last_map[i2+1] );
          MaxProp max_34 = std::max( last_map[i2+w2], last_map[i2+w2+1] );
          curr_map[i] = std::max( max_12, max_34 );
        }


        // check if greatest value it is a local max in the last level
        if ( curr_map[i].need_check )
        {
          double cv = curr_map[i].max_value;
          unsigned idx = curr_map[i].block_idx;

          bool is_local_max;

          if ( current_scale == 2 )
          {
            double* last_map = reinterpret_cast<double*>(img.data);
            is_local_max =
                cv > last_map[i2+diff[idx][0]] &&
                cv > last_map[i2+diff[idx][1]] &&
                cv > last_map[i2+diff[idx][2]] &&
                cv > last_map[i2+diff[idx][3]] &&
                cv > last_map[i2+diff[idx][4]];
          }
          else
          {
            is_local_max =
                cv > last_map[i2+diff[idx][0]].max_value &&
                cv > last_map[i2+diff[idx][1]].max_value &&
                cv > last_map[i2+diff[idx][2]].max_value &&
                cv > last_map[i2+diff[idx][3]].max_value &&
                cv > last_map[i2+diff[idx][4]].max_value;
          }

          if ( is_local_max )
          {
              curr_map[i].block_idx = x%2 + (y%2)*2;

              double s = (reinterpret_cast<double*>(scale_map.data))[curr_map[i].idx] * base_scale;

              if ( s <= s_thresh )
              {
                is_local_max = false;

                unsigned kp_x = curr_map[i].idx % img.cols;
                unsigned kp_y = curr_map[i].idx / img.cols;

                int window = s / 2;
                if ( window < 1 ) window = 1;
                window = 2*window + 1;

                //f << s << ", " << window << ", " << current_scale * 3 / 2 << std::endl;

                if (isnan( img[kp_y-1][kp_x-1] ) ||
                    isnan( img[kp_y-1][kp_x  ] ) ||
                    isnan( img[kp_y-1][kp_x+1] ) ||
                    isnan( img[kp_y  ][kp_x-1] ) ||
                    isnan( img[kp_y  ][kp_x  ] ) ||
                    isnan( img[kp_y  ][kp_x+1] ) ||
                    isnan( img[kp_y+1][kp_x-1] ) ||
                    isnan( img[kp_y+1][kp_x  ] ) ||
                    isnan( img[kp_y+1][kp_x+1] ))
                {

                } else if ( kp_x-s > 0 &&
                     kp_x+s < img.cols &&
                     kp_y-s > 0 &&
                     kp_y+s < img.rows )
                {
                  //std::cout<<kp_x<<" "<<kp_y<<" s="<<s<<" thresh="<<s_thresh<< std::endl;

                  // make keypoint
                  kp.push_back( KeyPoint ( kp_x, kp_y, s, -1, cv ) );
                }
              }
          }

          curr_map[i].need_check = is_local_max;
          if ( is_local_max )
          {
            remaining_checks++;
          }
        }
      }
    }

    //std::cout << "Remaining checks: " << remaining_checks << std::endl;

    if ( !remaining_checks )
    {
      break;
    }

    last_map.swap( curr_map );
  }
  //f.close();
  //std::cout << "Found " << kp.size() << " keypoints" << std::endl;
}


} 
