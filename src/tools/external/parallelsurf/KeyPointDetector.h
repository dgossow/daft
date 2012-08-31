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

#ifndef __parallelsurf_keypointdetector_h
#define __parallelsurf_keypointdetector_h

#include "Image.h"
#include "KeyPoint.h"

namespace parallelsurf
{

class KeyPointInsertor
{
	public:
		virtual void operator() ( const KeyPoint &k ) = 0;
};

class KeyPointDetector
{
	public:

		/**
		 * @brief default constructor
		 * @param iImage integral image to use
		 * @param iThreadPool Thread pool to use for computation
		 */
		KeyPointDetector();

		/// @brief set number of scales per octave
		inline void setMaxScales ( unsigned int iMaxScales )
		{
			_maxScales = iMaxScales;
		}

		/// @brief set number of octaves to search
		inline void setMaxOctaves ( unsigned int iMaxOctaves )
		{
			_maxOctaves = iMaxOctaves;
		}

		/// @brief set minimum threshold on determinant of hessian for detected maxima
		inline void setScoreThreshold ( double iThreshold )
		{
			_scoreThreshold = iThreshold;
		}

		/**
		 * @brief detect and store keypoints
		 * @param iImage integral image to use
		 * @param iInsertor function object used for storing the keypoints
		 */
		void detectKeyPoints ( Image& iImage, KeyPointInsertor& iInsertor );

	private:

		// internal values of the keypoint detector

		// number of scales
		int					_maxScales;

		// number of octaves
		int					_maxOctaves;

		// detection score threshold
		double							_scoreThreshold;

		// initial box filter size
		int					_initialBoxFilterSize;

		// scale overlapping : how many filter sizes to overlap
		// with default value 3 : [3,5,7,9,11][7,11,15,19,23][...
		int					_scaleOverlap;

		// some default values.
		const static double kBaseSigma;

		bool fineTuneExtrema ( double *** iSH, int iX, int iY, int iS,
		                       double& oX, double& oY, double& oS, double& oScore,
		                       int iOctaveWidth, int iOctaveHeight, int iBorder );

		bool calcTrace ( Image& iImage, double iX, double iY, double iScale, int& oTrace );

		int		getFilterSize ( int iOctave, int iScale );
		int		getBorderSize ( int iOctave, int iScale );

};

}

#endif //__parallelsurf_keypointdetector_h
