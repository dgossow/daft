//
//        _.---,._,'
//       /' _.--.<
//         /'     `'
//       /' _.---._____
//       \.'   ___, .-'`
//           /'    \\             .
//         /'       `-.          -|-
//        |                       |
//        |                   .-'~~~`-.
//        |                 .'         `.
//        |                 |  R  I  P  |
//        |                 |           |
//        |                 | Obsolete  |
//        |                 |   Code    |
//         \              \\|           |//
//   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


/** compute depth gradient
 * @param sp step width in projected pixel
 */
inline bool computeGradient(
    const Mat1f &depth_map,
    int x, int y, float sp, Vec2f& grad
) {
  // get depth values from image
  float d_center = depth_map(y,x);
  float d_xp = depth_map(y,x+sp);
  float d_yp = depth_map(y+sp,x);
  float d_xn = depth_map(y,x-sp);
  float d_yn = depth_map(y-sp,x);

  if ( isnan(d_center) || isnan(d_xp) || isnan(d_yp) || isnan(d_xn) || isnan(d_yn) )
  {
    grad[0]=0;
    grad[1]=0;
    return false;
  }

  float dxx = d_xp - 2*d_center + d_xn;
  float dyy = d_yp - 2*d_center + d_yn;

  const float cMaxCurvature = 2.0f;
  // test for local planarity
  // TODO note: this does not check for the case of a saddle
  if ( std::abs(dxx + dyy) > 2.0f*cMaxCurvature )
  {
    grad[0]=0;
    grad[1]=0;
    return false;
  }

// depth gradient between (x+sp) and (x-sp)
  grad[0] = (d_xp - d_xn)*0.5;
  grad[1] = (d_yp - d_yn)*0.5;
  return true;
}


// compute depth gradient
inline bool computeGradient2( const Mat1f &depth_map,
    int x, int y, float sp, float sw, Vec2f& grad )
{
  float d_center = depth_map(y,x);
  float d_right1 = depth_map(y,x+sp);
  float d_right2 = depth_map(y,x+sp*2);
  float d_left1 = depth_map(y,x-sp);
  float d_left2 = depth_map(y,x-sp*2);
  float d_bottom1 = depth_map(y+sp,x);
  float d_bottom2 = depth_map(y+sp*2,x);
  float d_top1 = depth_map(y-sp,x);
  float d_top2 = depth_map(y-sp*2,x);

  if ( isnan( d_center ) ||
      ( ( isnan(d_right1) || isnan(d_right2) ) && ( isnan(d_left1) || isnan(d_left2) ) ) ||
      ( ( isnan(d_top1) || isnan(d_top2) ) && ( isnan(d_bottom1) || isnan(d_bottom2) ) ) )
  {
    return false;
  }

  float dxx_right = std::abs( d_right2 - 2*d_right1 + d_center );
  float dxx_left = std::abs( d_center - 2*d_left1 + d_left2 );

  if ( isnan( dxx_right ) || dxx_left < dxx_right )
  {
    grad[0] = (d_center - d_left1);//2) * 0.5;
  }
  else
  {
    grad[0] = (d_right1 - d_center);
  }

  float dxx_top = std::abs( d_top2 - 2*d_top1 + d_center );
  float dxx_bottom = std::abs( d_center - 2*d_bottom1 + d_bottom2 );

  if ( isnan( dxx_top ) || dxx_bottom < dxx_top )
  {
    grad[1] = (d_bottom2 - d_center) * 0.5;
  }
  else
  {
    grad[1] = (d_center - d_top2) * 0.5;
  }

  assert( !isnan(grad[0]) && !isnan(grad[1]) );

  return true;
}



/* Compute Laplacian l(x,y) = f_xx(x,y)+f_yy(x,y)
 * Kernel size: 4s x 4s
 * Value range: 0..1
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  0  1  1  0  0
 *  0    0  1 -2 -2  1  0
 *  1    0  1 -2 -2  1  0
 *  2    0  0  1  1  0  0
 *  3    0  0  0  0  0  0
*/
inline double laplace( const Mat1d &ii, int x, int y, int s )
{
  if ( checkBounds( ii, x, y, 2*s ) )
  {
    double v = integral ( ii, x - 2*s, x + 2*s, y - s,   y + s   )
            +  integral ( ii, x - s,     x + s, y - 2*s, y + 2*s )
        -4.0 * integral ( ii, x - s,     x + s, y - s,   y + s   );
    return std::abs( v ) / (32.0*double(s*s));
  }
  return std::numeric_limits<double>::quiet_NaN();
}



/* Compute Harris corner measure h(x,y)
 * Value range: 0..1
*/
inline double harris( const Mat1d &ii, int x, int y, int s )
{
  if ( checkBounds( ii, x, y, 4*s ) )
  {
    double sum_dxdx=0;
    double sum_dydy=0;
    double sum_dxdy=0;

    // dx and dy have range -4s² .. 4s²
    double norm = 0.25 / double(s*s);
    int s2 = s*2;

    for ( int x2 = x-s; x2 <= x+s; x2 += s )
    {
      for ( int y2 = y-s; y2 <= y+s; y2 += s )
      {
        double dx = ( - integral ( ii, x2-s2, x2,    y2-s,  y2+s  )
                      + integral ( ii, x2,    x2+s2, y2-s,  y2+s  ) ) * norm;
        double dy = ( - integral ( ii, x2-s,  x2+s,  y2-s2, y2    )
                      + integral ( ii, x2-s,  x2+s,  y2,    y2+s2 ) ) * norm;
        sum_dxdx += dx * dx;
        sum_dydy += dy * dy;
        sum_dxdy += dx * dy;
      }
    }

    double trace = ( sum_dxdx + sum_dydy );
    double det = (sum_dxdx * sum_dydy) - (sum_dxdy * sum_dxdy);

    return det - 0.1 * (trace * trace);
  }

  return std::numeric_limits<double>::quiet_NaN();
}


/* Compute 2nd discrete derivative f_xx(x,y)
 * Result is not normalized!
 * Kernel size: 4s x 4s.
 * Value range: -8s²..8s²
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  0  0  0  0  0
 *  0    0  1 -1 -1  1  0
 *  1    0  1 -1 -1  1  0
 *  2    0  0  0  0  0  0
 *  3    0  0  0  0  0  0
*/
inline double dxx( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 2*s, x + 2*s, y - s, y + s )
  -2.0 * integral ( ii, x - s,   x + s,   y - s, y + s );
}

/* Compute 2nd discrete derivative f_yy(x,y)
 * Analogous to dxx(ii,x,y).
 */
inline double dyy( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - s, x + s, y - 2*s, y + 2*s )
  -2.0 * integral ( ii, x - s, x + s,   y - s, y + s   );
}




/* Compute 2nd discrete derivative f_xy(x,y)
 * Kernel size: 4s x 4s
 * Value range: -8s²..8s²
 * Kernel (s=1):
 *      -2 -1  0  1  2  3
 *
 * -2    0  0  0  0  0  0
 * -1    0  1  1 -1 -1  0
 *  0    0  1  1 -1 -1  0
 *  1    0 -1 -1  1  1  0
 *  2    0 -1 -1  1  1  0
 *  3    0  0  0  0  0  0
*/
inline double dxy( const Mat1d &ii, int x, int y, int s )
{
  return integral ( ii, x - 2*s, x, y - 2*s, y )
      - integral ( ii, x, x + 2*s, y - 2*s, y )
      - integral ( ii, x - 2*s, x, y, y + 2*s )
      + integral ( ii, x, x + 2*s, y, y + 2*s );
}

