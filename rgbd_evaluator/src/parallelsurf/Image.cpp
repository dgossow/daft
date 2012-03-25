/*
* This file is part of Parallel SURF, which implements the SURF algorithm
* using multi-threading.
*
* Copyright (C) 2010 David Gossow
*
* It is based on the SURF implementation included in Pan-o-matic 0.9.4,
* written by Anael Orlinski.
*
* Parallel SURF is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 3 of the License, or
* (at your option) any later version.
*
* Parallel SURF is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Image.h"

#include <omp.h>
#include <iostream>
#include <stdlib.h>

using namespace parallelsurf;

bool Image::DoRandomInit = false;

Image::Image ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight )
{
	init ( iPixels, iWidth, iHeight );
}

void Image::init ( const unsigned char **iPixels, unsigned int iWidth, unsigned int iHeight )
{
	// store values
	_width = iWidth;
	_height = iHeight;
	_pixels = iPixels;

	// allocate the integral image data
	_ii = AllocateImage ( _width + 1, _height + 1 );

	// create the integral image
	buildIntegralImage();
}

void Image::clean()
{
	if ( _ii )
		DeallocateImage ( _ii, _height + 1 );
	_ii = 0;
}


Image::~Image()
{
	clean();
}

void Image::buildIntegralImage()
{
	// to make easier the later computation, shift the image by 1 pix (x and y)
	// so the image has a size of +1 for width and height compared to orig image.

	// fill first line with zero
	for ( unsigned int i = 0; i <= _width; ++i )
		_ii[0][i] = 0;

	// fill first row with zero
	for ( unsigned int i = 0; i <= _height; ++i )
		_ii[i][0] = 0;

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
				_ii[i][j] = norm * double ( _pixels[i-1][j-1] ) + _ii[i][j-1];

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
				_ii[i][j] = norm * double ( _pixels[i-1][j-1] ) + _ii[i-1][j] + _ii[i][j-1] - _ii[i-1][j-1];
	}
}

// allocate and deallocate pixels
double** Image::AllocateImage ( unsigned int iWidth, unsigned int iHeight )
{
	// create the lines holder
	double ** aImagePtr = new double* [iHeight];

	// create the lines
	for ( unsigned int i = 0; i < iHeight; ++i )
		aImagePtr[i] = new double[iWidth];

	if ( DoRandomInit )
	{
		for ( unsigned int j = 0; j < iHeight; ++j )
			for ( unsigned int i = 0; i < iWidth; ++i )
				aImagePtr[j][i] = double(rand())/double(RAND_MAX)*1000000;
	}
	
	return aImagePtr;
}

void Image::DeallocateImage ( double **iImagePtr, unsigned int iHeight )
{
	// delete the lines
	for ( unsigned int i = 0; i < iHeight; ++i )
		delete[] iImagePtr[i];

	// delete the lines holder
	delete[] iImagePtr;

}

void Image::setDoRandomInit( bool iDoRandomInit )
{
	DoRandomInit = iDoRandomInit;
}