/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_hessian_filter_h
#define __rgbd_features_hessian_filter_h

#include "rgbd_features/math_stuff.h"

#ifndef CALC_INTEGRAL_SURFACE
#warning "CALC_INTEGRAL_SURFACE should really be defined!!"

#define CALC_INTEGRAL_SURFACE(II, STARTX, ENDX, STARTY, ENDY) \
  (II[ENDY][ENDX] + II[STARTY][STARTX] - II[ENDY][STARTX] - II[STARTY][ENDX])

#endif

namespace rgbd_features
{

class IntegralImage;

class HessianFilter
{
	public:
		
		HessianFilter ( IntegralImage& iImage );

		bool checkBounds ( int x, int y, int scale ) const;
		
		double getValue( int x, int y, int scale ) const;
		
		double getDxx( int x, int y, int scale ) const;
		double getDyy( int x, int y, int scale ) const;
		double getDxy( int x, int y, int scale ) const;

	private:
	
		// orig image info
		double**	_ii;
		unsigned int	_im_width;
		unsigned int	_im_height;
};

inline HessianFilter::HessianFilter ( IntegralImage& iImage )
{
	_ii = iImage.getIntegralImage();
	_im_width = iImage.getWidth();
	_im_height = iImage.getHeight();
}

inline double HessianFilter::getDxx( int x, int y, int scale ) const
{
	return	CALC_INTEGRAL_SURFACE ( _ii, x - 3*scale,	x + 3*scale, y - 2*scale, y + 2*scale )
	       - 3.0 * CALC_INTEGRAL_SURFACE ( _ii, x - scale,	x + scale, y - 2*scale, y + 2*scale );
}

inline double HessianFilter::getDyy( int x, int y, int scale ) const
{
	return	CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale, x + 2*scale, y - 3*scale, y + 3*scale )
	       - 3.0 * CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale,	x + 2*scale, y - scale, y + scale );
}

inline double HessianFilter::getDxy( int x, int y, int scale ) const
{
	return	 CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale, x, y - 2*scale, y )
				 - CALC_INTEGRAL_SURFACE ( _ii, x, x + 2*scale, y - 2*scale, y )
				 - CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale, x, y, y + 2*scale )
				 + CALC_INTEGRAL_SURFACE ( _ii, x, x + 2*scale, y, y + 2*scale );
}

inline double HessianFilter::getValue( int x, int y, int scale ) const
{
	double aDxy = getDxy( x, y, scale ) * 0.5;
	double aDxx = getDxx( x, y, scale );
	double aDyy = getDyy( x, y, scale );
	
	double c = 1.0 / double(scale*scale);

	return ( ( aDxx * aDyy ) - ( aDxy * aDxy ) ) * c * c;
}

inline bool HessianFilter::checkBounds ( int x, int y, int scale ) const
{
	return (    x > 3*scale && x + 3*scale < ( int ) _im_width
	         &&	y > 3*scale && y + 3*scale < ( int ) _im_height );
}

} // namespace rgbd_features

#endif //__rgbd_features_hessian_filter_h
