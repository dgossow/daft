/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_harrisfilter_h
#define __rgbd_features_harrisfilter_h

#include "math_stuff.h"

namespace rgbd_features
{

template<int Size>
class HarrisFilter
{
	public:
		
		HarrisFilter<Size> ( IntegralImage& iImage );

		bool checkBounds ( int x, int y, int scale ) const;
		
		double getValue( int x, int y, int scale ) const;
		
    double getDx( int x, int y, int scale ) const;
    double getDy( int x, int y, int scale ) const;

	private:
	
		// orig image info
		double**	_ii;
		unsigned int	_im_width;
		unsigned int	_im_height;
};

template<int Size>
inline HarrisFilter<Size>::HarrisFilter ( IntegralImage& iImage )
{
	_ii = iImage.getIntegralImage();
	_im_width = iImage.getWidth();
	_im_height = iImage.getHeight();
}

template<int Size>
inline double HarrisFilter<Size>::getDx( int x, int y, int scale ) const
{
  return - CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale,  x, y - scale, y + scale )
         + CALC_INTEGRAL_SURFACE ( _ii, x,  x + 2*scale, y - scale, y + scale );
}

template<int Size>
inline double HarrisFilter<Size>::getDy( int x, int y, int scale ) const
{
  return - CALC_INTEGRAL_SURFACE ( _ii, x - scale,  x + scale, y - 2*scale, y )
         + CALC_INTEGRAL_SURFACE ( _ii, x - scale,  x + scale, y, y + 2*scale );
}

template<int Size>
inline double HarrisFilter<Size>::getValue( int x, int y, int scale ) const
{
  double sum_dx2=0;
  double sum_dy2=0;
  double sum_dxdy=0;

  double norm = 1.0 / double(scale*scale);
//  double norm2 = norm * norm;

  for ( int x_shifted = x-scale*Size; x_shifted <= x+scale*Size; x_shifted += scale )
  {
    for ( int y_shifted = y-scale*Size; y_shifted <= y+scale*Size; y_shifted += scale )
    {
      double dx = getDx( x_shifted, y_shifted, scale ) * norm;
      double dy = getDy( x_shifted, y_shifted, scale ) * norm;
      sum_dx2 += dx *dx;
      sum_dy2 += dy * dy;
      sum_dxdy += dx * dy;
    }
  }

  double trace = ( sum_dx2 + sum_dy2 );
  double det = (sum_dx2 * sum_dy2) - (sum_dxdy * sum_dxdy);

  return det - 0.1 * (trace * trace);
}

template<int Size>
inline bool HarrisFilter<Size>::checkBounds ( int x, int y, int scale ) const
{
	return (    x > (Size+2)*scale && x + (Size+2)*scale < ( int ) _im_width
	         &&	y > (Size+2)*scale && y + (Size+2)*scale < ( int ) _im_height );
}

} // namespace rgbd_features

#endif //__rgbd_features_boxfilter_h
