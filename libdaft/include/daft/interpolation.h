/*
 * interpolation.h
 *
 *  Created on: Jul 4, 2012
 *      Author: gossow
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_

#include <opencv2/core/core.hpp>

#include <cmath>

namespace cv
{
namespace daft
{

// interpolation functions
namespace inter
{
  template<typename S> inline S linear( S x ){ return x; }
  template<typename S> inline S smooth( S x ) { return 0.5*(1.0-cos(x*M_PI)); }
  template<typename S> inline S nearest( S x ) { return round(x); }
  template<typename S> inline S zero( S x ) { return 0; }
  template<typename S> inline S one( S x ) { return 1; }
}

template<typename S, typename T, S (*F)(S x)>
inline T interp( T y1, T y2, S t )
{
  const S t2 = F( t );
  return (1.0-t2) * y1 + t2 * y2;
}

template<typename S, typename T, S (*F)(S x)>
inline T interp2d( S v00, S v01, S v10, S v11, S tx, S ty )
{
  const T v1 = interp<S,T,F>( v00, v01, tx );
  const T v2 = interp<S,T,F>( v10, v11, tx );
  const T v = interp<S,T,F>( v1, v2, ty );
  return v;
}

template<typename S, typename T, S (*F)(S x)>
inline T interp2d( const Mat_<T>& img, S x, S y )
{
  //x-=0.5;
  //y-=0.5;
  if ( x < 0 ) x = 0;
  if ( y < 0 ) y = 0;

  int x_low = x;
  int y_low = y;
  int x_high = x_low + 1;
  int y_high = y_low + 1;

  if ( x >= img.cols-1 )
  {
    x_low = x_high = x = img.cols-1;
  }
  if ( y >= img.rows-1 )
  {
    y_low = y_high = y = img.rows-1;
  }

  const S tx = x - S(x_low);
  const S ty = y - S(y_low);
  return interp2d<S,T,F>(
      img(y_low,x_low), img(y_low,x_high),
      img(y_high,x_low), img(y_high,x_high),
      tx, ty );
  assert(1);
}

// 3d-interpolation in mipmap, using the interpolation functions
// F in image space (x,y) and G between levels of detail (lod)
template<typename S, typename T, S (*F)(S x), S (*G)(S x)>
inline T interpMipMap( const std::vector< Mat_<T> >& mipmaps, S x, S y, S lod )
{
  if ( lod < 0.0 ) lod = 0.0;
  if ( lod >= mipmaps.size()-1 ) return mipmaps[mipmaps.size()-1](0,0);

  const int lod1 = lod;
  const int pow2l = 1 << lod1;

  float o1 = (1.0-pow2l) / (2.0*pow2l);

  const S x1 = (float)x/((float)pow2l) + o1;
  const S y1 = (float)y/((float)pow2l) + o1;
  const S x2 = (x1*0.5) - 0.25;
  const S y2 = (y1*0.5) - 0.25;

  const S v1 = interp2d<S,T,F >( mipmaps[lod1], x1, y1 );
  const S v2 = interp2d<S,T,F >( mipmaps[lod1+1], x2, y2 );

  //return interp< S,T,inter::linear<S> >( v1, v2, std::pow( 2.0, lod - (S)lod1 ) * 0.5 );
  return interp< S,T,G >( v1, v2, lod - (S)lod1 );
}

}
}

#endif /* INTERPOLATION_H_ */
