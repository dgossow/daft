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

#ifndef __parallelsurf_keypointdescriptorcontext_h
#define __parallelsurf_keypointdescriptorcontext_h

namespace parallelsurf
{
class KeyPoint;

/**
* @brief Helper construct used by KeyPointDescriptor
*/
class KeyPointDescriptorContext
{
	public:
		KeyPointDescriptorContext ( int iSubRegions, int iVecLen, double iOrientation );
		~KeyPointDescriptorContext();

		/// @brief number of square subregions (1 direction)
		int    _subRegions; 

		/// @brief sinus of orientation
		double   _sin;
		
		/// @brief cosinus of orientation
		double   _cos;

		/// @brief descriptor components
		double***  _cmp;

		void placeInIndex ( double iMag1, int iOri1, double iMag2, int iOri2, double iUIdx, double iVIdx );

		void placeInIndex2 ( double iMag1, int iOri1, double iUIdx, double iVIdx );

	private:
		// disallow stupid things
		KeyPointDescriptorContext();
		KeyPointDescriptorContext ( const KeyPointDescriptorContext& );
		KeyPointDescriptorContext& operator= ( KeyPointDescriptorContext& ) throw();

};

}

#endif //__parallelsurf_keypointdescriptorcontext_h
