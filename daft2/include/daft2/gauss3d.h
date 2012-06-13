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

//#define DBG_WIN

inline float gauss2(float sigma, float d2) {
  return 0.39894228f / sigma * std::exp(-0.5f * d2 / (sigma * sigma));
}

inline int makePow2( double x )
{
  return std::pow( 2.0, std::ceil( log2( x ) ) );
}

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
  //cv::imshow("tmp",tmp);

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

    double n = mipmaps.size() - 1;

    /*
    int rows = std::ceil(img.rows / std::pow(2,n));
    int cols = std::ceil(img.cols / std::pow(2,n));
    cv::Mat1f dbg_img;
    resize< float,float,inter::nearest<float> >( cv::Mat1f(tmp,cv::Rect(0,0,rows,cols)), dbg_img, img.rows, img.cols );
    std::stringstream s;
    s << "mipmaps[" << mipmaps.size()-1 << "] rows=" << tmp.rows << " cols=" << tmp.cols;
    imshow( s.str(), dbg_img );
    */
    //imshow2( s.str()+" 2", tmp, img.rows );

    mipmap_offs.push_back( (1.0-std::pow(2.0,n)) / (std::pow(2.0,n+1)) );
  }

  cv::waitKey(100);

  const int lut_size = 100;
  const float lut_1 = 35;
  float gaussval[lut_size];

  for (int i=0; i<lut_size; i++ )
  {
    float x2 = (float)i / (float)lut_1;
    const float sigma = 0.735534255;
    gaussval[i] = gauss2( sigma, x2 );
    //std::cout << "x²=" << x2 << " << g(x)=" << gaussval[i] << std::endl;
  }
  //std::cout << std::endl;

  Mat_<Point3f> xyz_map( img.rows, img.cols );

  for ( int i = 0; i < img.rows; i++ )
  {
    for ( int j = 0; j < img.cols; ++j )
    {
      float z = depth_map(i,j);
      getPt3d( f_inv, cx, cy, j, i, z, xyz_map(i,j) );
    }
  }

  //const float base_scale2 = base_scale*base_scale;

  boost::timer t;

  cv::Mat3b lod_img( img_out.rows, img_out.cols, cv::Vec3b(0,255,255) );

  for ( int i = 0; i < img.rows; i++ )
  {
    for ( int j = 0; j < img.cols; ++j )
    {
      if ( !(depth_map(i,j) > 0.0))
      {
        img_out(i,j) = std::numeric_limits<float>::quiet_NaN();
        continue;
      }

      //float win = 1.5 * base_scale * f / depth_map(i,j) + 1.0;
      float winf = 1.5 * base_scale * f / depth_map(i,j);
      if ( winf < 0.5 ) winf=0.5;
      int num_steps = winf;
      const int max_num_steps = 8;
      if ( num_steps > max_num_steps ) num_steps = max_num_steps;

      float lodf = log2( winf / (float)(num_steps) );
      if ( lodf < 0 ) lodf = 0;
      int lod = lodf;

      /*
      if ( i==100 && j == 100 )
      {
        std::cout << "lod " << lod << std::endl;
        cv::Mat1f l1,l2;
        resize< float,float,inter::nearest<float> >( mipmaps[lod], l1, img.rows, img.cols );
        resize< float,float,inter::nearest<float> >( mipmaps[lod+1], l2, img.rows, img.cols );
        imshow( "mipmaps[lod]", mipmaps[lod] );
        imshow( "mipmaps[lod+1]", mipmaps[lod+1] );
      }
      */

      const int pow2l = 1 << lod;

      //std::cout << pow2l / (winf / (float)(num_steps)) << std::endl;

      Point3f c = xyz_map(i,j);

      float sum_val = 0.0;
      float sum_weight = 0.0;

#ifdef DBG_WIN
      cv::Mat1f w_img( num_steps*2+1, num_steps*2+1, 1.0f );
#endif

      //img_out(i,j) = interpMipMap< float,float,inter::linear<float> >( mipmaps, j, i, lodf );
      //continue;

      //lod_img(i,j)[0] = lod*30;
      //lod_img(i,j)[2] = img(i,j);

      for ( int u = -num_steps; u <= num_steps; u++ )
      {
        int i2 = i + u * winf / (float)(num_steps);
        if ( i2 > img.rows-2 ) i2 = img.rows-2;
        if ( i2 < 0 ) i2 = 0;

        for ( int v = -num_steps; v <= num_steps; v++ )
        {
          int j2 = j + v * winf / (float)(num_steps);
          if ( j2 > img.cols-2 ) j2 = img.cols-2;
          if ( j2 < 0 ) j2 = 0;

          if ( depth_map(i2,j2) > 0 )
          {
            Point3f d = (xyz_map(i2,j2) - c) * (1.0 / base_scale);
            float d2 = ( d.x*d.x+d.y*d.y+d.z*d.z );
            int idx = std::min( lut_size-1, (int)(d2*lut_1) );
            //float w = d2 < 1.0 ? 1.0 : 0.0;// (idx < lut_size) ? gaussval[idx] : 0.0;
            float w = (idx >= 0 && idx < lut_size) ? gaussval[idx] : 0.0;
            sum_weight += w;
            float j_mm = (float)j2/(float)(pow2l-1) + mipmap_offs[lod];
            float i_mm = (float)i2/(float)(pow2l-1) + mipmap_offs[lod];
            //sum_val += w * interp2d< float,float,inter::nearest<float> >( mipmaps[lod], j_mm, i_mm );
            sum_val += w * interpMipMap< float,float,inter::linear<float> >( mipmaps, j2, i2, lodf );

#ifdef DBG_WIN
            if ( j==320 && i==240 )
            {
              //std::cout << sqrt(d2) << "  " << w << std::endl;
              w_img( u+num_steps, v+num_steps ) = w;
            }
#endif
          }
        }
        //assert(jc == num_steps+1);
      }
      //assert(ic == num_steps+1);

#ifdef DBG_WIN
      std::ostringstream s;
      s << base_scale;
      if ( j==320 && i==240 )
      {
        imshowNorm( s.str( ), w_img );
        std::cout << "size ratio: " << pow2l / (winf / (float)(num_steps)) << std::endl;
      }
#endif

      img_out(i,j) = sum_val / sum_weight;
    }

    /*
    if ( t.elapsed() > 1.0 )
    {
      cv::imshow("img_out",img_out);
      cv::waitKey(50);
      t.restart();
    }
    */
  }

  cvtColor(lod_img, lod_img, CV_HSV2BGR );
  //cv::imshow("lod_img",lod_img);

  //cv::imshow("img_out",img_out);
  cv::waitKey(50);
}


}
}

#endif /* GAUSS3D_H_ */
