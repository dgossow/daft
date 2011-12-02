/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_dobfilter_h
#define __rgbd_features_dobfilter_h

#include "rgbd_features/math_stuff.h"
#include "rgbd_features/integral_image.h"

namespace rgbd_features
{

// Implements a difference-of-boxes filter, approximating difference-of-gaussians
template<int Size>
class DobFilter
{
	public:
		
		DobFilter ( IntegralImage& iImage );

		bool checkBounds ( int x, int y, int scale ) const;
		
		double getValue( int x, int y, int scale ) const;
		
	private:
	
		// orig image info
		double**	_ii;
		unsigned int	_im_width;
		unsigned int	_im_height;

		static const double OUTER_FACTOR = 1.0 / (Size*Size);
};

template<int Size>
DobFilter<Size>::DobFilter ( IntegralImage& iImage )
{
	_ii = iImage.getIntegralImage();
	_im_width = iImage.getWidth();
	_im_height = iImage.getHeight();
}

template<int Size>
inline double DobFilter<Size>::getValue( int x, int y, int scale ) const
{
  int size = Size*scale;

  double val = CALC_INTEGRAL_SURFACE ( _ii, x - scale,  x + scale, y - scale, y + scale )
      - OUTER_FACTOR * CALC_INTEGRAL_SURFACE ( _ii, x - size, x + size, y - size, y + size );

  double val_norm = val / double(size*size);
  return val_norm * val_norm * 10;
}

template<int Size>
inline bool DobFilter<Size>::checkBounds ( int x, int y, int scale ) const
{
	return (    x > Size*scale && x + Size*scale < ( int ) _im_width
	         &&	y > Size*scale && y + Size*scale < ( int ) _im_height );
}

} // namespace rgbd_features

#endif
