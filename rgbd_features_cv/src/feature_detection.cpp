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
    std::vector< KeyPoint3D >& kp )
{
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

      // Round scale, substract the one extra pixel we have in the center
      int window = s; //(s+0.5) / 2.0 - 0.5;
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
        kp.push_back( cv::KeyPoint3D ( x, y, s*2, base_scale, -1, img[y][x] ) );
      }
    }
  }
}




// Helper struct + comparison
struct MaxProp
{
  // function value
  float max_value;

  // index inside (last) 2x2 block. Layout:
  // 0 1
  // 2 3
  unsigned int block_idx;

  // index in original image
  unsigned int idx;

  // still needs check for local max
  bool need_check;

  inline operator float() const
  {
    return max_value;
  }
};

template<typename T>
inline unsigned int maxIdx(const T& x0, const T& x1, const T& x2, const T& x3)
{
  return x0>x1
      ? ( x2>x3 ? (x0>x2?0:2) : (x0>x3?0:3) )
      : ( x2>x3 ? (x1>x2?1:2) : (x1>x3?1:3) );
}

template<typename T>
inline bool isLocalMax( float cv, const T& max_map,
    unsigned max_idx, bool win_5x5, unsigned w )
{
  if ( win_5x5 )
  {
    return
        cv > max_map[max_idx-2*w-1] &&
        cv > max_map[max_idx-2*w] &&
        cv > max_map[max_idx-2*w+1] &&
        cv > max_map[max_idx-w-2] &&
        cv > max_map[max_idx-w-1] &&
        cv > max_map[max_idx-w] &&
        cv > max_map[max_idx-w+1] &&
        cv > max_map[max_idx-w+2] &&
        cv > max_map[max_idx-2] &&
        cv > max_map[max_idx-1] &&
        cv > max_map[max_idx+1] &&
        cv > max_map[max_idx+2] &&
        cv > max_map[max_idx+w-2] &&
        cv > max_map[max_idx+w-1] &&
        cv > max_map[max_idx+w] &&
        cv > max_map[max_idx+w+1] &&
        cv > max_map[max_idx+w+2] &&
        cv > max_map[max_idx+2*w-1] &&
        cv > max_map[max_idx+2*w] &&
        cv > max_map[max_idx+2*w+1];
  }
  else
  {
    return
        cv > max_map[max_idx-w-1] &&
        cv > max_map[max_idx-w] &&
        cv > max_map[max_idx-w+1] &&
        cv > max_map[max_idx-1] &&
        cv > max_map[max_idx+1] &&
        cv > max_map[max_idx+w-1] &&
        cv > max_map[max_idx+w] &&
        cv > max_map[max_idx+w+1];
  }
}

//define DGB_F

void findMaximaMipMap( const cv::Mat1d &img,
    const cv::Mat1d &scale_map,
    double base_scale,
    double thresh,
    std::vector< KeyPoint3D >& kp )
{
  if ( !img.isContinuous() )
  {
    return;
  }

  double max_dim = std::max( img.rows, img.cols );

  unsigned w_next = img.cols;
  unsigned h_next = img.rows;
  unsigned next_map_size = img.rows*img.cols;

  std::vector< MaxProp > max_map;

#ifdef DGB_F
  std::ofstream f;
  f.open( "/tmp/wt2.csv" );
#endif

  // compute max levels
  for ( int px_size = 1; px_size<max_dim; px_size*=2 )
  {
    const unsigned w = w_next;
    const unsigned h = h_next;
    w_next /= 2;
    h_next /= 2;
    next_map_size = w_next*h_next;

    if ( w_next<3 || h_next<3 ) break;

    const int next_px_size = px_size * 2;

    // the next map is going to be half as wide/tall as the current
    std::vector< MaxProp > next_max_map( next_map_size );

    // fixed value for rounding to next scale level
    // the factor is computed to minimize the error in
    // search area: sqrt(5/2) * 1.5
    //float s_thresh_old = float(next_px_size) * 1.41 * 1.5;

    // above this threshold. take a 5x5 instead of a 3x3 neighbourhood
    float s_thresh = float(px_size*3) * 0.889756521;
    float s_thresh_2 = float(px_size*3) * 0.645497224;

    //std::cout << "Mipmap level " << current_scale << " thresh " << s_thresh << " size " << w << " x " << h << std::endl;

    const int block_idx_offset[4] = { 0, 1, w, w + 1};

    unsigned remaining_checks = 0;

    for ( unsigned y_next = 0; y_next < h_next; y_next++ )
    {
      for ( unsigned x_next = 0; x_next < w_next; ++x_next )
      {
        unsigned x = 2*x_next;
        unsigned y = 2*y_next;

        unsigned i_next = x_next + y_next*w_next;
        unsigned i = x + y*w;

        unsigned max_blockidx,max_idx;

        // compute max values in 2x2 blocks of higher level
        if ( px_size == 1 )
        {
          double* max_map = reinterpret_cast<double*>(img.data);
          max_blockidx = maxIdx(max_map[i], max_map[i+1], max_map[i+w], max_map[i+w+1]);
          MaxProp& p = next_max_map[i_next];
          max_idx = i + block_idx_offset[max_blockidx];
          p.idx = max_idx;
          p.max_value = max_map[max_idx];
          p.need_check = p.max_value > thresh;
        }
        else
        {
          max_blockidx = maxIdx(max_map[i], max_map[i+1], max_map[i+w], max_map[i+w+1]);
          max_idx = i + block_idx_offset[max_blockidx];
          next_max_map[i_next] = max_map[max_idx];
        }

        if ( next_max_map[i_next].need_check )
        {
          // if we have reached the nearest level to the actual scale,
          // check for local maximum
          double s = (reinterpret_cast<double*>(scale_map.data))[next_max_map[i_next].idx] * base_scale;

          if ( s <= s_thresh )
          {
#ifdef DGB_F
            int old_window = s*0.5; //(s+0.5) / 2.0 - 0.5;
            if ( old_window < 1 ) old_window = 1;
            // this is the real side length of the window used.
            old_window = 2*old_window + 1;

            float new_window = s > s_thresh_2 ? 4.58 * float(px_size) : 3 * float(px_size);

            f << s << ", " << old_window << ", " << new_window << std::endl;
#endif

            next_max_map[i_next].need_check = false;

            if ( ( x > 2 ) && ( y > 2 ) && ( x < w-2 ) && ( y < h-2 ) )
            {
              float cv = next_max_map[i_next].max_value;

              bool is_local_max;

              if ( px_size == 1 )
              {
//                inline bool isLocalMax( double cv, const T& max_map,
//                    unsigned max_idx, bool win_5x5, unsigned w )
                double* max_map = reinterpret_cast<double*>(img.data);
                is_local_max = isLocalMax( cv, max_map, max_idx, s > s_thresh_2, w );
              }
              else
              {
                is_local_max = isLocalMax( cv, max_map, max_idx, s > s_thresh_2, w );
              }

              if ( is_local_max )
              {
                unsigned kp_x = next_max_map[i_next].idx % img.cols;
                unsigned kp_y = next_max_map[i_next].idx / img.cols;

                if (finite( img[kp_y-1][kp_x-1] ) &&
                    finite( img[kp_y-1][kp_x  ] ) &&
                    finite( img[kp_y-1][kp_x+1] ) &&
                    finite( img[kp_y  ][kp_x-1] ) &&
                    finite( img[kp_y  ][kp_x  ] ) &&
                    finite( img[kp_y  ][kp_x+1] ) &&
                    finite( img[kp_y+1][kp_x-1] ) &&
                    finite( img[kp_y+1][kp_x  ] ) &&
                    finite( img[kp_y+1][kp_x+1] ))
                {
                  // make keypoint
                  kp.push_back( KeyPoint3D ( kp_x, kp_y, s*2, base_scale, -1, cv ) );
                }
              }
            }
          }
          else
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

    max_map.swap( next_max_map );
  }

  //std::cout << "Found " << kp.size() << " keypoints" << std::endl;
}

} 
