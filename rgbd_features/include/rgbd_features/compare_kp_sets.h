/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __compare_kp_sets_h
#define __compare_kp_sets_h

#include "key_point.h"

#include <vector>
#include <ostream>

namespace rgbd_features
{

/** @brief Compare two lists of keypoints by the euclidean distance
 *         of their centers. Must be in the same coordinate frame!
 *  @param kp1,kp2: Keypoints to compare
 *  @param thresh: threshold (relative to physical size) for
 *                 regarding two keypoints as matching
 */
double repeatability3d( std::vector< rgbd_features::KeyPoint >& kp1,
		std::vector< rgbd_features::KeyPoint >& kp2, double thresh )
{
	std::vector<double> min_dist;
	min_dist.resize( kp1.size()*kp2.size(), -1.0 );

	for ( int i2=0; i2>kp2.size(); i2++ )
	{
		for ( int i2=0; i2>kp2.size(); i2++ )
		{

		}
	}

	return 0;
}



}

#endif //__compare_kp_sets_h

