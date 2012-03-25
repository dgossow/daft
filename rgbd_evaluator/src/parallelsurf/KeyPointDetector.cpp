/*
* Copyright (C) 2007-2008 Anael Orlinski
*
* This file is part of Panomatic.
*
* Panomatic is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* Panomatic is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with Panomatic; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <iostream>

#include "KeyPoint.h"
#include "KeyPointDetector.h"
#include "BoxFilter.h"
#include "MathStuff.h"
#include <stdlib.h>
#include <omp.h>

using namespace parallelsurf;

const double parallelsurf::KeyPointDetector::kBaseSigma = 1.2;

KeyPointDetector::KeyPointDetector()
{
	// initialize default values
	_maxScales = 5;		// number of scales : 9x9, 15x15, 21x21, 27x27, ...
	_maxOctaves = 4;	// number of octaves

	_scoreThreshold = 0.1;

	_initialBoxFilterSize = 3;
	_scaleOverlap = 3;
}

void KeyPointDetector::detectKeyPoints ( Image& iImage, KeyPointInsertor& iInsertor )
{
	// allocate lots of memory for the scales
	double *** aSH = new double**[_maxScales];
	for ( int s = 0; s < _maxScales; ++s )
		aSH[s] = Image::AllocateImage ( iImage.getWidth(), iImage.getHeight() );

	// init the border size
	int * aBorderSize = new int[_maxScales];

	int aMaxima = 0;

	// base size + 3 times first increment for step back
	// for the first octave 9x9, 15x15, 21x21, 27x27, 33x33
	// for the second 21x21, 33x33, 45x45 ...

	// go through all the octaves
	for ( int o = 0; o < _maxOctaves; ++o )
	{
		// calculate the pixel step on the image, and the image size
		int aPixelStep = 1 << o;	// 2^aOctaveIt
		int aOctaveWidth = iImage.getWidth() / aPixelStep;	// integer division
		int aOctaveHeight = iImage.getHeight() / aPixelStep;	// integer division

		// fill each scale matrices
		for ( int s = 0; s < _maxScales; ++s )
		{
			// calculate the border for this scale
			aBorderSize[s] = getBorderSize ( o, s );

			// create a box filter of the correct size for each thread
			std::vector< BoxFilter > aBoxFilters;
			aBoxFilters.reserve( omp_get_max_threads() );
			for ( int i=0; i<omp_get_max_threads(); ++i )
			{
				aBoxFilters.push_back( BoxFilter( getFilterSize ( o, s ), iImage ) );
			}

			// fill the hessians
			int aEy = aOctaveHeight - 2 * aBorderSize[s];
			int aEx = aOctaveWidth - aBorderSize[s];

			int yIt;

			//initialize the memory with random values (only used for testing/debugging)
 			if ( Image::getDoRandomInit() )
			{
#pragma omp parallel for
				for ( int y = 0; y < aOctaveHeight; y++ )
				{
					for ( int x = 0; x < aOctaveWidth; ++x )
					{
						aSH[s][y][x] = double(rand())/double(RAND_MAX)*1000000;
					}
				}
			}
			
#pragma omp parallel for
			for ( yIt = 0; yIt < aEy; yIt++ )
			{
				int y = aBorderSize[s] + yIt;
				int aYPS = y * aPixelStep;
				int t = omp_get_thread_num();
				
				aBoxFilters[ t ].setY ( aYPS );

				int aXPS = aBorderSize[s] * aPixelStep;
				for ( int x = aBorderSize[s]; x < aEx; ++x )
				{
					aSH[s][y][x] = aBoxFilters[ t ].getDetWithX ( aXPS );
					aXPS += aPixelStep;
				}
			}
		}

		// detect the feature points with a 3x3x3 neighborhood non-maxima suppression
		for ( int aSIt = 1; aSIt < ( _maxScales - 1 ); aSIt += 2 )
		{
			int aBS = aBorderSize[ aSIt+1 ];
			
 			int aNextScale = aSIt < _maxScales - 2 ? aSIt+2 : aSIt+1;
			int aNextBS = aBorderSize[ aNextScale ];
			
#pragma omp parallel for
			for ( int aYIt = aBS + 1; aYIt < aOctaveHeight - aBS - 1; aYIt += 2 )
			{
				for ( int aXIt = aBS + 1; aXIt < aOctaveWidth - aBS - 1; aXIt += 2 )
				{
					// find the maximum in the 2x2x2 cube
					double aTab[8];

					// get the values in a
					aTab[0] = aSH[aSIt]  [aYIt]  [aXIt];
					aTab[1] = aSH[aSIt]  [aYIt]  [aXIt+1];
					aTab[2] = aSH[aSIt]  [aYIt+1][aXIt];
					aTab[3] = aSH[aSIt]  [aYIt+1][aXIt+1];
					aTab[4] = aSH[aSIt+1][aYIt]  [aXIt];
					aTab[5] = aSH[aSIt+1][aYIt]  [aXIt+1];
					aTab[6] = aSH[aSIt+1][aYIt+1][aXIt];
					aTab[7] = aSH[aSIt+1][aYIt+1][aXIt+1];

					// find the max index without using a loop.
					int a04 = ( aTab[0] > aTab[4] ? 0 : 4 );
					int a15 = ( aTab[1] > aTab[5] ? 1 : 5 );
					int a26 = ( aTab[2] > aTab[6] ? 2 : 6 );
					int a37 = ( aTab[3] > aTab[7] ? 3 : 7 );
					int a0426 = ( aTab[a04] > aTab[a26] ? a04 : a26 );
					int a1537 = ( aTab[a15] > aTab[a37] ? a15 : a37 );
					int aMaxIdx = ( aTab[a0426] > aTab[a1537] ? a0426 : a1537 );

					// calculate approximate threshold
					double aApproxThres = _scoreThreshold * 0.8;

					double aScore = aTab[aMaxIdx];

					// check found point against threshold
					if ( aScore < aApproxThres )
						continue;

					// verify that other missing points in the 3x3x3 cube are also below threshold
					int aXShift = 2 * ( aMaxIdx & 1 ) - 1;
					int aXAdj = aXIt + ( aMaxIdx & 1 );
					aMaxIdx >>= 1;

					int aYShift = 2 * ( aMaxIdx & 1 ) - 1;
					int aYAdj = aYIt + ( aMaxIdx & 1 );
					aMaxIdx >>= 1;

					int aSShift = 2 * ( aMaxIdx & 1 ) - 1;
					int aSAdj = aSIt + ( aMaxIdx & 1 );

					// skip too high scale ajusting
					if ( aSAdj == ( int ) _maxScales - 1 )
						continue;
					
					//if we adjusted the scale to aSIt+1, then we also have to check 
					//for the border size of the next higher scale
					if ( ( aSShift == 1 ) &&
						   ( ( aXAdj < aNextBS + 1 ) || 
							   ( aXAdj >= aOctaveWidth - aNextBS - 1 ) ||
						     ( aYAdj < aNextBS + 1 ) || 
							   ( aYAdj >= aOctaveHeight - aNextBS - 1 ) ) 
							)
					{
						continue;
					}

					if (	( aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj - 1] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj    ] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj - aYShift][aXAdj + 1] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj			][aXAdj - 1] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj			][aXAdj    ] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj			][aXAdj + 1] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj - 1] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj    ] > aScore ) ||
					     ( aSH[aSAdj + aSShift][aYAdj + aYShift][aXAdj + 1] > aScore ) ||

					     ( aSH[aSAdj][		aYAdj + aYShift	][aXAdj - 1] > aScore ) ||
					     ( aSH[aSAdj][			aYAdj + aYShift	][aXAdj] > aScore ) ||
					     ( aSH[aSAdj][		aYAdj + aYShift	][aXAdj + 1] > aScore ) ||
					     ( aSH[aSAdj][	aYAdj			][aXAdj + aXShift] > aScore ) ||
					     ( aSH[aSAdj][	aYAdj - aYShift	][aXAdj + aXShift] > aScore ) ||

					     ( aSH[aSAdj - aSShift][		aYAdj + aYShift	][aXAdj - 1] > aScore ) ||
					     ( aSH[aSAdj - aSShift][			aYAdj + aYShift	][aXAdj] > aScore ) ||
					     ( aSH[aSAdj - aSShift][		aYAdj + aYShift	][aXAdj + 1] > aScore ) ||
					     ( aSH[aSAdj - aSShift][	aYAdj			][aXAdj + aXShift] > aScore ) ||
					     ( aSH[aSAdj - aSShift][	aYAdj - aYShift	][aXAdj + aXShift] > aScore )
					   )
						continue;

					// fine tune the location
					double aX = aXAdj;
					double aY = aYAdj;
					double aS = aSAdj;

					// try to fine tune, restore the values if it failed
					// if the returned value is true,  keep the point, else drop it.
					if ( !fineTuneExtrema ( aSH, aXAdj, aYAdj, aSAdj, aX, aY, aS, aScore, aOctaveWidth, aOctaveHeight, aBorderSize[aSAdj] ) )
						continue;

					// recheck the updated score
					if ( aScore < _scoreThreshold )
						continue;

					// adjust the values
					aX *= aPixelStep;
					aY *= aPixelStep;
					aS = ( ( 2 * aS * aPixelStep ) + _initialBoxFilterSize + ( aPixelStep - 1 ) * _maxScales ) / 3.0; // this one was hard to guess...

					// store the point
					int aTrace;
					if ( !calcTrace ( iImage, aX, aY, aS, aTrace ) )
						continue;

					aMaxima++;

#pragma omp critical
					{
						// do something with the keypoint depending on the insertor
						iInsertor ( KeyPoint ( aX, aY, aS * kBaseSigma, aScore, aTrace ) );
					}

				}
			}
		}
	}

	// deallocate memory of the scale images
	for ( int s = 0; s < _maxScales; ++s )
		Image::DeallocateImage ( aSH[s], iImage.getHeight() );
}

bool KeyPointDetector::fineTuneExtrema ( double *** iSH, int iX, int iY, int iS,
  double& oX, double& oY, double& oS, double& oScore,
  int iOctaveWidth, int iOctaveHeight, int iBorder )
{
	// maximum fine tune iterations
	const int	kMaxFineTuneIters = 6;

	// shift from the initial position for X and Y (only -1 or + 1 during the iterations).
	int aX = iX;
	int aY = iY;
	int aS = iS;

	int aShiftX = 0;
	int aShiftY = 0;

	// current deviations
	double aDx = 0, aDy = 0, aDs = 0;

	//result vector
	double aV[3];	//(x,y,s)

	for ( int aIter = 0; aIter < kMaxFineTuneIters; ++aIter )
	{
		// update the extrema position
		aX += aShiftX;
		aY += aShiftY;

		// create the problem matrix
		double aM[3][3]; //symetric, no ordering problem.

		// fill the result vector with gradient from pixels differences (negate to prepare system solve)
		aDx = ( iSH[aS  ][aY  ][aX+1] - iSH[aS  ][aY  ][aX-1] ) * 0.5;
		aDy = ( iSH[aS  ][aY+1][aX  ] - iSH[aS  ][aY-1][aX  ] ) * 0.5;
		aDs = ( iSH[aS+1][aY  ][aX  ] - iSH[aS-1][aY  ][aX  ] ) * 0.5;

		aV[0] = - aDx;
		aV[1] = - aDy;
		aV[2] = - aDs;

		// fill the matrix with values of the hessian from pixel differences
		aM[0][0] = iSH[aS  ][aY  ][aX-1] - 2.0 * iSH[aS][aY][aX] + iSH[aS  ][aY  ][aX+1];
		aM[1][1] = iSH[aS  ][aY-1][aX  ] - 2.0 * iSH[aS][aY][aX] + iSH[aS  ][aY+1][aX  ];
		aM[2][2] = iSH[aS-1][aY  ][aX  ] - 2.0 * iSH[aS][aY][aX] + iSH[aS+1][aY  ][aX  ];

		aM[0][1] = aM[1][0] = ( iSH[aS  ][aY+1][aX+1] + iSH[aS  ][aY-1][aX-1] - iSH[aS  ][aY+1][aX-1] - iSH[aS  ][aY-1][aX+1] ) * 0.25;
		aM[0][2] = aM[2][0] = ( iSH[aS+1][aY  ][aX+1] + iSH[aS-1][aY  ][aX-1] - iSH[aS+1][aY  ][aX-1] - iSH[aS-1][aY  ][aX+1] ) * 0.25;
		aM[1][2] = aM[2][1] = ( iSH[aS+1][aY+1][aX  ] + iSH[aS-1][aY-1][aX  ] - iSH[aS+1][aY-1][aX  ] - iSH[aS-1][aY+1][aX  ] ) * 0.25;

		// solve the linear system. results are in aV. exit with error if a problem happened
		if ( !Math::SolveLinearSystem33 ( aV, aM ) )
			return false;


		// ajust the shifts with the results and stop if no significant change

		if ( aIter < kMaxFineTuneIters - 1 )
		{
			aShiftX = 0;
			aShiftY = 0;

			if ( aV[0] > 0.6 && aX < ( int ) ( iOctaveWidth - iBorder - 2 ) )
				aShiftX++;
			else if ( aV[0] < -0.6 && aX > ( int ) iBorder + 1 )
				aShiftX--;

			if ( aV[1] > 0.6 && aY < ( int ) ( iOctaveHeight - iBorder - 2 ) )
				aShiftY++;
			else if ( aV[1] < -0.6 && aY > ( int ) iBorder + 1 )
				aShiftY--;

			if ( aShiftX == 0 && aShiftY == 0 )
				break;
		}
	}

	// update the score
	oScore = iSH[aS][aY][aX] + 0.5 * ( aDx * aV[0] + aDy * aV[1] + aDs * aV[2] );

	// reject too big deviation in last step (unfinished job).
	if ( Math::Abs ( aV[0] ) > 1.5 || Math::Abs ( aV[1] ) > 1.5  || Math::Abs ( aV[2] ) > 1.5 )
		return false;

	// put the last deviation (not integer :) to the output
	oX = aX + aV[0];
	oY = aY + aV[1];
	oS = iS + aV[2];

	return true;
}

int	KeyPointDetector::getFilterSize ( int iOctave, int iScale )
{
	int aScaleShift = 2 << iOctave;
	return	_initialBoxFilterSize + ( aScaleShift - 2 ) * ( _maxScales - _scaleOverlap ) + aScaleShift * iScale;
}

int	KeyPointDetector::getBorderSize ( int iOctave, int iScale )
{
	int aScaleShift = 2 << iOctave;
	if ( iScale <= 2 )
	{
		int aMult = ( iOctave == 0 ? 1 : 2 );
		return ( getFilterSize ( iOctave, 1 ) + aMult * aScaleShift ) * 3 / aScaleShift + 1;
	}
	return getFilterSize ( iOctave, iScale ) * 3 / aScaleShift + 1;
}

bool KeyPointDetector::calcTrace ( Image& iImage,
                                   double iX,
                                   double iY,
                                   double iScale,
                                   int& oTrace )
{
	int aRX = Math::Round ( iX );
	int aRY = Math::Round ( iY );

	BoxFilter aBox ( 3*iScale, iImage );

	if ( !aBox.checkBounds ( aRX, aRY ) )
		return false;

	aBox.setY ( aRY );
	double aTrace = aBox.getDxxWithX ( aRX ) + aBox.getDyyWithX ( aRX );
	oTrace = ( aTrace <= 0.0 ? -1 : 1 );

	return true;
}

