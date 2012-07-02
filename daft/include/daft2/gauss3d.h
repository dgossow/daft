/*
 * gauss3d.h
 *
 *  Created on: Jun 8, 2012
 *      Author: gossow
 */

#ifndef GAUSS3D_H_
#define GAUSS3D_H_

#include <opencv2/core/core.hpp>

#include <boost/timer.hpp>

#include <math.h>
#include <cmath>

#include "stuff.h"

namespace cv {
namespace daft2 {

inline float gauss2(float sigma, float d2) {
  return 0.39894228f / sigma * std::exp(-0.5f * d2 / (sigma * sigma));
}

inline int makePow2( double x )
{
  return std::pow( 2.0, std::ceil( log2( x ) ) );
}

template<typename S, typename T, S (*F)(S x), S (*G)(S x)>
inline void gauss3d( cv::Matx33f K,
    const Mat1f &depth_map,
    const Mat1b &img,
    float base_scale,
    Mat1f &img_out )
{
  img_out = Mat1f( img.rows, img.cols, 0.0f );

  float f = K(0,0);
  float f_inv = 1.0 / f;
  float cx = K(0, 2);
  float cy = K(1, 2);

  std::vector< cv::Mat1f > mipmaps;
  std::vector< float > mipmap_offs;

  cv::Mat1f tmp( makePow2(img.rows), makePow2(img.cols), 0.0f );
  for ( int i = 0; i < tmp.rows; i++ )
  {
    int i2 = i < img.rows ? i : 2*img.rows-1-i;
    for ( int j = 0; j < tmp.cols; ++j )
    {
      int j2 = j < img.cols? j : 2*img.cols-1-j;
      tmp(i,j) = (float)img(i2,j2) / 255.0;
    }
  }

  mipmaps.push_back( tmp.clone() );
  mipmap_offs.push_back(0.0);

  while(tmp.rows > 1 && tmp.cols > 1)
  {
    cv::Mat1f tmp2( tmp.rows/2, tmp.cols/2 );
    for ( int i = 0; i < tmp2.rows; i++ )
    {
      for ( int j = 0; j < tmp2.cols; ++j )
      {
        tmp2(i,j) = ( tmp(i*2,j*2)+tmp(i*2,j*2+1)+tmp(i*2+1,j*2)+tmp(i*2+1,j*2+1) ) * 0.25;
      }
    }

    tmp=tmp2;
    mipmaps.push_back( tmp.clone() );

    /*
    double n = mipmaps.size() - 1;

    int rows = std::ceil(img.rows / std::pow(2,n));
    int cols = std::ceil(img.cols / std::pow(2,n));
    cv::Mat1f dbg_img;
    resize< S,T,F >( cv::Mat1f(tmp,cv::Rect(0,0,rows,cols)), dbg_img, img.rows, img.cols );
    //cv::Mat1f dbg_img2;
    //resize< S,T,F >( cv::Mat1f(tmp,cv::Rect(0,0,rows,cols)), dbg_img2, img.rows, img.cols );
    std::stringstream s;
    s << "mipmaps[" << mipmaps.size()-1 << "] rows=" << tmp.rows << " cols=" << tmp.cols;
    //imshowNorm( s.str(), dbg_img+(dbg_img2*0.1), -1 );
    imshowNorm( s.str(), dbg_img, -1 );
    //imshow2( s.str()+" 2", tmp, img.rows );
     */
  }

  const int lut_size = 100;
  const float lut_1 = 30;
  float gaussval[lut_size];

  for (int i=0; i<lut_size; i++ )
  {
    float x2 = (float)i / (float)lut_1;
    const float sigma = 0.735534255;
    gaussval[i] = gauss2( sigma, x2 );
  }

  // experimentally determined:
  base_scale *= 52.0/64.0;

  Mat_<Point3f> xyz_map( img.rows, img.cols );

  for ( int i = 0; i < img.rows; i++ )
  {
    for ( int j = 0; j < img.cols; ++j )
    {
      float z = depth_map(i,j);
      getPt3d( f_inv, cx, cy, j, i, z, xyz_map(i,j) );
    }
  }

  for ( int i = 0; i < img.rows; i++ )
  {
    for ( int j = 0; j < img.cols; ++j )
    {
      if ( !(depth_map(i,j) > 0.0))
      {
        img_out(i,j) = std::numeric_limits<float>::quiet_NaN();
        continue;
      }

      float winf = 2.0 * base_scale * f / depth_map(i,j);
      const float desired_num_steps = 6;

      float lodf = log2( std::sqrt(2.0) * winf / desired_num_steps );
      if ( lodf < 0.0 ) lodf = 0.0;

      int lod = lodf;
      const int step_size = 1 << lod;

      const int num_steps = round( winf / (float)step_size );

      Point3f c = xyz_map(i,j);

      /*
      if ( i==0 && j== 0 )
      {
        std::cout << "lodf = " << lodf << " num_steps = " << num_steps << std::endl;
      }
      */

      float sum_val0 = 0.0;
      float sum_val1 = 0.0;
      float sum_weight0 = 0.0;
      float sum_weight1 = 0.0;

      //img_out(i,j) = interpMipMap< float,float,inter::linear<float> >( mipmaps, j, i, lodf );
      //continue;

      for ( int u = -num_steps; u <= num_steps; u++ )
      {
        int i2 = i + u * step_size;
        if ( i2 > img.rows-2 ) i2 = img.rows-2;
        if ( i2 < 0 ) i2 = 0;

        for ( int v = -num_steps; v <= num_steps; v++ )
        {
          int j2 = j + v * step_size;
          if ( j2 > img.cols-2 ) j2 = img.cols-2;
          if ( j2 < 0 ) j2 = 0;

          if ( depth_map(i2,j2) > 0 )
          {
            Point3f d = (xyz_map(i2,j2) - c) * (1.0 / base_scale);
            float d2 = ( d.x*d.x+d.y*d.y+d.z*d.z );
            int idx = d2*lut_1;//std::min( lut_size-1, (int)(d2*lut_1) );
            float w = (idx >= 0 && idx < lut_size) ? gaussval[idx] : 0.0;

            sum_weight0 += w;
            sum_val0 += w * interpMipMap< S,T,F, inter::zero<float> >( mipmaps, j2, i2, lod );
            if ( u%2==0 && v%2==0 )
            {
              sum_weight1 += w;
              sum_val1 += w * interpMipMap< S,T,F, inter::zero<float> >( mipmaps, j2, i2, lod+1 );
            }

          }
        }
      }

      float v0 = sum_val0 / sum_weight0;
      float v1 = sum_val1 / sum_weight1;
      float t = lodf - (float)lod;
      img_out(i,j) = interp< S,T,G >( v0, v1, t );
    }
  }
}


}
}

#endif /* GAUSS3D_H_ */
