/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_keypointdetector_h
#define __rgbd_features_keypointdetector_h

#include "rgbd_features/integral_image.h"
#include "rgbd_features/key_point.h"

#include <limits>
#include <assert.h>
#include <math.h>

#define USE_LINEAR_SCALE_INTERPOLATION

namespace rgbd_features
{

template <class FilterT>
void filter( IntegralImage& iImage,
             double** iScaleImage,
             double iScale,
             double** oFilteredImage );

void findMaxima( IntegralImage& iImage,
                 double** iScaleImage,
                 double iScale,
                 double** iFilteredImage,
                 std::vector< rgbd_features::KeyPoint >& oKp,
                 double iThresh = 0.1 );

template <class FilterT>
void filterMaxima( IntegralImage& iImage,
                   double** iScaleImage,
                   double iScale,
                   double** iFilteredImage,
                   std::vector< rgbd_features::KeyPoint >& ioKp,
                   double iThresh = 0.1 );

///throw away keypoints near depth edges or areas with missing depth information
template <int WindowSize>
void filterByDepthEdges( double** iScaleImage,
                         std::vector< rgbd_features::KeyPoint >& ioKp,
                         double iThresh );

bool calcTrace ( IntegralImage& oFilteredImage,
                 double iX,
                 double iY,
                 double iScale,
                 int& oTrace );

//----------------------------------------------------------

template <class FilterT>
void filter( rgbd_features::IntegralImage& iImage,
             double** iScaleImage,
             double iScale,
             double** oFilteredImage )
{
    FilterT filter( iImage );
//#pragma omp parallel for
    for ( unsigned y = 0; y < iImage.getHeight(); y++ )
    {
        for ( unsigned x = 0; x < iImage.getWidth(); ++x )
        {
            double scale = iScaleImage[y][x] * iScale;// * 0.25 - 1;
            if ( scale <= 2.0 )
            {
                oFilteredImage[y][x] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            if ( !filter.checkBounds ( x, y, scale+1 ) )
            {
                oFilteredImage[y][x] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

#ifdef USE_LINEAR_SCALE_INTERPOLATION
            float t = scale - floor(scale);
            oFilteredImage[y][x] = (1.0-t) * filter.getValue( x, y, int(scale) )
                + t * filter.getValue( x, y, int(scale)+1 );
#else
            oFilteredImage[y][x] = filter.getValue( x, y, scale+0.5 );
#endif
        }
    }
}


template <class FilterT>
void filterMaxima( IntegralImage& iImage,
                   double** iFilteredImage,
                   std::vector< rgbd_features::KeyPoint >& ioKp,
                   double iThresh = 0.1 )
{
  std::vector< rgbd_features::KeyPoint > kp = ioKp;
  ioKp.clear();
  ioKp.reserve( kp.size() );

  FilterT filter( iImage );

  for ( unsigned k=0; k<kp.size(); k++ )
  {
    int x = kp[k]._x;
    int y = kp[k]._y;

    double image_scale = kp[k]._image_scale;
    if ( image_scale <= 1.0 || !filter.checkBounds ( x, y, ceil(image_scale) ) )
    {
        continue;
    }

    float t = image_scale - floor(image_scale);
    double response = (1.0-t) * filter.getValue( x, y, floor(image_scale) )
        + t * filter.getValue( x, y, ceil(image_scale) );

    if ( response > iThresh )
    {
      ioKp.push_back( kp[k] );
    }
  }
}

}

#undef USE_LINEAR_SCALE_INTERPOLATION

#endif //__rgbd_features_keypointdetector_h
