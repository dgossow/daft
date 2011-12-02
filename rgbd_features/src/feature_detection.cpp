/*
* Copyright (C) 2011 David Gossow
*/

#include <iostream>

#include "rgbd_features/key_point.h"
#include "rgbd_features/integral_image.h"
#include "rgbd_features/hessian_filter.h"
#include "rgbd_features/feature_detection.h"
#include "rgbd_features/math_stuff.h"

#include <stdlib.h>
#include <stdio.h>

#include <math.h>

namespace rgbd_features
{
	

void findMaxima( rgbd_features::IntegralImage& iImage,
                 double** iScaleImage,
                 double iScale,
                 double** iFilteredImage,
                 std::vector< rgbd_features::KeyPoint >& oKp,
                 double iThresh )
{
	//find maxima in sxs neighbourhood
#pragma omp parallel for
	for ( unsigned y = 3; y < iImage.getHeight()-3; y++ )
	{
		for ( unsigned x = 3; x < iImage.getWidth()-3; ++x )
		{
			if ( iFilteredImage[y][x] < iThresh || isnan( iFilteredImage[y][x] ) )
			{
				continue;
			}
			
      double image_scale = iScaleImage[y][x] * iScale;// * 0.25 - 1;
			
			if ( x-image_scale < 0 || x+image_scale >= iImage.getWidth() || y-image_scale < 0 || y+image_scale > iImage.getHeight() )
			{
				continue;
			}
			
      if (isnan( iFilteredImage[y-1][x-1] ) ||
          isnan( iFilteredImage[y  ][x+1] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] ) ||
          isnan( iFilteredImage[y+1][x  ] )
          )
      {
        continue;
      }

			int window = image_scale / 2;
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
					if ( ( iFilteredImage[y+v][x+u] >= iFilteredImage[y][x] ) ||
					     ( iFilteredImage[y+v][x-u] >= iFilteredImage[y][x] ) ||
					     ( iFilteredImage[y-v][x+u] >= iFilteredImage[y][x] ) ||
					     ( iFilteredImage[y-v][x-u] >= iFilteredImage[y][x] ) )
					{
						isMax=false;
					}
				}
			}
			
			if ( isMax )
			{
				int trace = 1;
				if ( calcTrace ( iImage, x, y, image_scale, trace ) )
				{
#pragma omp critical
				  oKp.push_back ( KeyPoint ( x, y, image_scale, iScale, iFilteredImage[y][x], trace ) );
				}
			}
		}
	}
}


bool calcTrace ( IntegralImage& oFilteredImage,
                 double iX,
                 double iY,
                 double iScale,
                 int& oTrace )
{
	int aRX = Math::Round ( iX );
	int aRY = Math::Round ( iY );

	HessianFilter aBox ( oFilteredImage );

	if ( !aBox.checkBounds ( aRX, aRY, iScale ) )
		return false;

	double aTrace = aBox.getDxx( aRX, aRY, iScale ) + aBox.getDxx( aRX, aRY, iScale );
	oTrace = ( aTrace <= 0.0 ? -1 : 1 );

	return true;
}


} 
