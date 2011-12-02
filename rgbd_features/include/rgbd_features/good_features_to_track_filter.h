/*
* Copyright (C) 2011 David Gossow
*/

#ifndef __rgbd_features_good_features_to_track_filter_h
#define __rgbd_features_good_features_to_track_filter_h

#include <math.h>

#include "rgbd_features/math_stuff.h"

namespace rgbd_features
{

template<int Size>
class GoodFeaturesToTrackFilter
{
	public:
		
		GoodFeaturesToTrackFilter<Size> ( IntegralImage& iImage );

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
inline GoodFeaturesToTrackFilter<Size>::GoodFeaturesToTrackFilter ( IntegralImage& iImage )
{
	_ii = iImage.getIntegralImage();
	_im_width = iImage.getWidth();
	_im_height = iImage.getHeight();
}

template<int Size>
inline double GoodFeaturesToTrackFilter<Size>::getDx( int x, int y, int scale ) const
{
  return - CALC_INTEGRAL_SURFACE ( _ii, x - 2*scale,  x, y - scale, y + scale )
         + CALC_INTEGRAL_SURFACE ( _ii, x,  x + 2*scale, y - scale, y + scale );
}

template<int Size>
inline double GoodFeaturesToTrackFilter<Size>::getDy( int x, int y, int scale ) const
{
  return - CALC_INTEGRAL_SURFACE ( _ii, x - scale,  x + scale, y - 2*scale, y )
         + CALC_INTEGRAL_SURFACE ( _ii, x - scale,  x + scale, y, y + 2*scale );
}

template<int Size>
inline double GoodFeaturesToTrackFilter<Size>::getValue( int x, int y, int scale ) const
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

  double S = sqrt( trace*trace*0.25 - det );
  double lambda1 = trace/2 + S;
  double lambda2 = trace/2 - S;

  return lambda1 < lambda2 ? lambda1 : lambda2;
}

template<int Size>
inline bool GoodFeaturesToTrackFilter<Size>::checkBounds ( int x, int y, int scale ) const
{
	return (    x > (Size+2)*scale && x + (Size+2)*scale < ( int ) _im_width
	         &&	y > (Size+2)*scale && y + (Size+2)*scale < ( int ) _im_height );
}


/*

// https://github.com/AndreaCensi/2x2_matrix_eigenvalues/blob/master/2x2_eigenvalues.c
#define tolerance 0.1e-20

void solve_eig(double A, double B, double C, double D,
    double &lambda1, double &lambda2, )
{
    if(B*C <= tolerance  ) {
        lambda1 = A;
        lambda2 = D;
        return;
    }

    double tr = A + D;
    double det = A * D - B * C;
    double S = sqrt( square(tr/2) - det );
    lambda1 = tr/2 + S;
    lambda2 = tr/2 - S;
}

 */


} // namespace rgbd_features

#endif //__rgbd_features_good_features_to_track_filter_h
