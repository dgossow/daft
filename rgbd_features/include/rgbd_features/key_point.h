/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_keypoint_h
#define __rgbd_features_keypoint_h

#include <vector>
#include <ostream>

namespace rgbd_features
{

class KeyPoint
{
	public:
		KeyPoint();
		KeyPoint ( double x, double y, double image_scale, double physical_scale, double score, int trace );

		double _x, _y;
		double _image_scale;
    double _physical_scale;
		double _score;
		int _trace;
		double _ori;

		double _rx,_ry,_rz;

		std::vector<double> _vec;
};

inline KeyPoint::KeyPoint()
{

}

inline KeyPoint::KeyPoint ( double x, double y, double image_scale, double physical_scale, double score, int trace ) :
		_x ( x ), _y ( y ), _image_scale ( image_scale ), _physical_scale ( physical_scale ), _score ( score ), _trace ( trace ), _vec ( 0 )
{
}

inline bool operator < ( const KeyPoint & iA, const KeyPoint & iB )
{
	return ( iA._score < iB._score );
}

}

#endif //__rgbd_features_keypoint_h

