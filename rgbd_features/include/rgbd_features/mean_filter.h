/*
* Copyright (C) 2011 David Gossow
*/

#ifndef rgbd_features_meanfilter_h_
#define rgbd_features_meanfilter_h_

#include "rgbd_features/math_stuff.h"

#include <limits>

namespace rgbd_features
{

class IntegralImage;

class MeanFilter
{
	public:
		
		MeanFilter ( IntegralImage& iImage );

		bool checkBounds ( int x, int y, int scale ) const;
		
		double getValue( int x, int y, int scale ) const;
		
	private:
	
		// orig image info
		double**	_ii;
		unsigned int	_im_width;
		unsigned int	_im_height;
};

inline MeanFilter::MeanFilter ( IntegralImage& iImage )
{
	_ii = iImage.getIntegralImage();
	_im_width = iImage.getWidth();
	_im_height = iImage.getHeight();
}

inline double MeanFilter::getValue( int x, int y, int scale ) const
{
	return CALC_INTEGRAL_SURFACE ( _ii, x - scale,	x + scale, y - scale, y + scale ) / (scale*scale);
}

#undef CALC_INTEGRAL_SURFACE

inline bool MeanFilter::checkBounds ( int x, int y, int scale ) const
{
	if ( scale < 0.0 )
	{
		scale = 0.0;
	}
	int scale2 = scale+1;
	return (    x > scale2 && x + scale2 < ( int ) _im_width
	         &&	y > scale2 && y + scale2 < ( int ) _im_height );
}

} // namespace rgbd_features

#endif //rgbd_features_meanfilter_h_
