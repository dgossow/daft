/*
* Copyright (C) 2011 David Gossow
*/

#include "rgbd_features/integral_image.h"

#include <omp.h>
#include <iostream>
#include <stdlib.h>

using namespace rgbd_features;

IntegralImage::IntegralImage ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight )
{
	init ( iWidth, iHeight );
	computeFrom( iPixels );
}

void IntegralImage::init ( unsigned int iWidth, unsigned int iHeight )
{
	// store values
	_width = iWidth;
	_height = iHeight;

	// allocate the integral image data
	_ii = AllocateImage ( _width + 1, _height + 1 );
	
	// fill first line/row with zero
	for ( unsigned int i = 0; i <= _width; ++i )
		_ii[0][i] = 0;
	for ( unsigned int i = 0; i <= _height; ++i )
		_ii[i][0] = 0;
}

void IntegralImage::clean()
{
	if ( _ii )
		DeallocateImage ( _ii, _height + 1 );
	_ii = 0;
}


IntegralImage::~IntegralImage()
{
	clean();
}

void IntegralImage::computeFrom(const unsigned char** iPixels)
{
	// to make easier the later computation, shift the image by 1 pix (x and y)
	// so the image has a size of +1 for width and height compared to orig image.
	// first row/col is 0
	
	static const double norm = 1.0 / 255.0;

	// compute all the others pixels

	int numCPUs = omp_get_num_procs();
	int maxThreads = omp_get_max_threads();
	
	if ( ( numCPUs > 2 ) && ( maxThreads > 2 ) )
	{
		//implementation suited for parallelized computation
		//only faster on 3 or more processors
#pragma omp parallel for
		//calculate row sums
		for ( unsigned int i = 1; i <= _height; ++i )
			for ( unsigned int j = 1; j <= _width; ++j )
				_ii[i][j] = norm * double ( iPixels[i-1][j-1] ) + _ii[i][j-1];

#pragma omp parallel for
		//sum up along columns
		for ( unsigned int j = 1; j <= _width; ++j )
			for ( unsigned int i = 1; i <= _height; ++i )
				_ii[i][j] += _ii[i-1][j];
	}
	else
	{
		for ( unsigned int i = 1; i <= _height; ++i )
			for ( unsigned int j = 1; j <= _width; ++j )
				_ii[i][j] = norm * double ( iPixels[i-1][j-1] ) + _ii[i-1][j] + _ii[i][j-1] - _ii[i-1][j-1];
	}
}

// allocate and deallocate pixels
double** IntegralImage::AllocateImage ( unsigned int iWidth, unsigned int iHeight )
{
	// create the lines holder
	double ** aImagePtr = new double* [iHeight];

	// create the lines
	for ( unsigned int i = 0; i < iHeight; ++i )
		aImagePtr[i] = new double[iWidth];

	return aImagePtr;
}

void IntegralImage::DeallocateImage ( double **iImagePtr, unsigned int iHeight )
{
  if ( !iImagePtr ) return;

	// delete the lines
	for ( unsigned int i = 0; i < iHeight; ++i )
		delete[] iImagePtr[i];

	// delete the lines holder
	delete[] iImagePtr;

}
