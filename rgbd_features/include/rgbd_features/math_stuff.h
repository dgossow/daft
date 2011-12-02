/*
* Copyright (C) 2011 David Gossow
*/

#define CALC_INTEGRAL_SURFACE(II, STARTX, ENDX, STARTY, ENDY) \
  (II[ENDY][ENDX] + II[STARTY][STARTX] - II[ENDY][STARTX] - II[STARTY][ENDX])

#define PI	3.14159

#ifndef rgbd_features_math_stuff_h_
#define rgbd_features_math_stuff_h_

#include <vector>

namespace rgbd_features
{
struct Math
{

	static bool				SolveLinearSystem33 ( double *solution, double sq[3][3] );
	static inline double	Abs ( const double iD )
	{
		return ( iD > 0.0 ? iD : -iD );
	}
	static inline int		Round ( const double iD )
	{
		return ( int ) ( iD + 0.5 );
	}
	static bool				Normalize ( std::vector<double> &iVec );


};

template < int LBound = -128, int UBound = 127, class TResult = double, class TArg = double >
class LUT
{
	public:
		explicit LUT ( TResult ( *f ) ( TArg ), double coeffadd = 0, double coeffmul = 1 )
		{
			lut = lut_array - LBound;
			for ( int i = LBound; i <= UBound; i++ )
			{
				lut[i] = f ( coeffmul * ( i + coeffadd ) );
			}
		}

		const TResult & operator() ( int i ) const
		{
			return lut[i];
		}
	private:
		TResult lut_array[UBound - LBound + 1];
		TResult * lut;
};

}

#endif //rgbd_features_math_stuff_h_
