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

template<typename K>
inline float LocalFiniteDifferencesKinect(K v0, K v1, K v2, K v3, K v4)
{
  if(isnan(v0) && isnan(v4) && !isnan(v1) && !isnan(v3)) {
    return float(v3 - v1);
  }

  bool left_invalid = (isnan(v0) || isnan(v1));
  bool right_invalid = (isnan(v3) || isnan(v4));
  if(left_invalid && right_invalid) {
    return 0.0f;
  }
  else if(left_invalid) {
    return float(v4 - v2);
  }
  else if(right_invalid) {
    return float(v2 - v0);
  }
  else {
    float a = static_cast<float>(std::abs(v2 + v0 - static_cast<K>(2)*v1));
    float b = static_cast<float>(std::abs(v4 + v2 - static_cast<K>(2)*v3));
    float p, q;
    if(a + b == 0.0f) {
      p = q = 0.5f;
    }
    else {
      p = a / (a + b);
      q = b / (a + b);
    }
    return q * static_cast<float>(v2 - v0) + p * static_cast<float>(v4 - v2);
  }
}


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



















#if 0
  Mat1f patch( DescPatchSize+1, DescPatchSize+1, nan );

#ifdef DESC_DEBUG_IMG
  Mat1f patch_dx( DescPatchSize, DescPatchSize, nan );
  Mat1f patch_dy( DescPatchSize, DescPatchSize, nan );
  Mat1f patch_weights( DescPatchSize, DescPatchSize, 1.0f );
  static const int PATCH_MUL = 16;
  Mat1f points3d_img( DescPatchSize*PATCH_MUL, DescPatchSize*PATCH_MUL, 1.0f );

  for ( int v=0; v<points3d_img.rows; v++ )
  {
    for ( int u=0; u<points3d_img.rows; u++ )
    {
      if ( ( (v+1+DescPatchSize*PATCH_MUL/10) % (DescPatchSize*PATCH_MUL/5) == 0 ) ||
           ( (u+1+DescPatchSize*PATCH_MUL/10) % (DescPatchSize*PATCH_MUL/5) == 0 ) )
      {
        points3d_img[v][u] = 0.75;
      }
    }
  }

  Mat1f points3d_rot_img = points3d_img.clone();
  Mat3b display_img;
  cvtColor( smoothed_img, display_img, CV_GRAY2BGR );
#endif

  //const Point2f& pt
  float angle = kp.affine_angle;
  float major = kp.affine_major*0.5;
  float minor = kp.affine_minor*0.5;

  // camera params
  float f_inv = 1.0 / K(0,0);
  float cx = K(0,2);
  float cy = K(1,2);

  Point3f affine_major_axis( 0,-1,0 );//cos(angle), sin(angle), 0 );

  Vec3f n = getNormal(kp, depth_map, K );

  Point3f normal(n[0],n[1],n[2]);
  Point3f v1 = normal.cross( affine_major_axis );
  v1 = v1  * fastInverseLen( v1 );
  Point3f v2 = v1.cross( normal );

  // transforms from local 3d coords [-0.5...0.5] to 3d
  cv::Matx33f local_to_cam( -v1.x, -v2.x, normal.x, -v1.y, -v2.y, normal.y, -v1.z, -v2.z, normal.z );

  // transforms from u/v texture coords [-PatchSize/2 ... PatchSize/2] to 3d
  cv::Matx33f uvw_to_cam = local_to_cam * kp.world_size * 0.5;

  // transforms from 3d to u/v tex coords
  cv::Matx33f cam_to_uvw = cv::Matx33f(2.0 / kp.world_size,0,0, 0,2.0 / kp.world_size,0, 0,0,1) * local_to_cam.t();

  // sample intensity values using planar assumption
  for ( int v = 0; v<DescPatchSize+1; v++ )
  {
    for ( int u = 0; u<DescPatchSize+1; u++ )
    {
      // compute u-v coordinates
      Point2f uv( float(u)-float(DescPatchSize/2), float(v-float(DescPatchSize/2)) );

      const float dist2 = (uv.x*uv.x + uv.y*uv.y);
      if ( dist2 > float((DescPatchSize+1)*(DescPatchSize+1)) * 0.25f )
      {
        continue;
      }

      Point2f pixel;
      Point3f pt_cam = (uvw_to_cam * Point3f(uv.x,uv.y,0)) + kp.pt3d;
      getPt2d( pt_cam, K(0,0), cx, cy, pixel );

      if ( checkBounds( smoothed_img, pixel.x, pixel.y, 1 ) )
      {
        patch[v][u] = interpBilinear(smoothed_img,pixel.x,pixel.y);

#ifdef DESC_DEBUG_IMG
        if ( !isnan( patch[v][u] ) )
        {
          float s = 0.5 * kp.world_size * K(0,0) / depth_map(int(pixel.y),int(pixel.x));
          Size2f bsize( s, minor/major*s );
          cv::RotatedRect box(pixel, bsize, angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,0,255), 1, CV_AA );
        }
#endif
      }
    }
  }


#ifdef DESC_DEBUG_IMG
  {
    Point3f pn3d = kp.pt3d + (normal*kp.world_size);
    Point3f pv13d = kp.pt3d + (v1*kp.world_size);
    Point3f pv23d = kp.pt3d + (v2*kp.world_size);

    Point2f p,pn,pv1,pv2;

    getPt2d( pn3d, (float)K(0,0), cx, cy, pn );
    getPt2d( pv13d, (float)K(0,0), cx, cy, pv1 );
    getPt2d( pv23d, (float)K(0,0), cx, cy, pv2 );
    getPt2d( kp.pt3d, (float)K(0,0), cx, cy, p );

    cv::line( display_img, p, pv1, cv::Scalar( 0,0,255 ), 2, 16 );
    cv::line( display_img, p, pv2, cv::Scalar( 0,255,0 ), 2, 16 );
    cv::line( display_img, p, pn, cv::Scalar( 255,0,0 ), 2, 16 );
  }
#endif

  vector<PtInfo> pt_infos;
  pt_infos.reserve(DescPatchSize*DescPatchSize);

  // compute gradients
  for ( int v = 0; v<DescPatchSize; v++ )
  {
    for ( int u = 0; u<DescPatchSize; u++ )
    {
      // 0-centered u/v coords
      Point2f uv( float(u)-center_uv, float(v)-center_uv );

      Point2f pixel;
      Point3f pt_cam = (uvw_to_cam * Point3f(uv.x,uv.y,0)) + kp.pt3d;
      getPt2d( pt_cam, K(0,0), cx, cy, pixel );

      if ( !checkBounds( smoothed_img, pixel.x, pixel.y, 1 ) )
      {
        continue;
      }

      // get the 3d coordinates of the points
      Point3f pt3d;
      int pixel_x_int = pixel.x + 0.5;
      int pixel_y_int = pixel.y + 0.5;
      getPt3d( f_inv, cx, cy, pixel.x, pixel.y, depth_map[pixel_y_int][pixel_x_int], pt3d );

      // local 3d tex coords of point [-PatchSize/2,PatchSize/2]
      Point3f pt3d_uvw = cam_to_uvw * (pt3d - kp.pt3d);

      // uv indices of reprojected 3d point [0,PatchSize]
      Point2f uv_reproj( pt3d_uvw.x + float(DescPatchSize/2), pt3d_uvw.y + float(DescPatchSize/2) );

      // normalized patch coords [-1,1]
      Point3f pt3d_uvw1 = pt3d_uvw * (2.0f/float(DescPatchSize));
      float dist_2 = pt3d_uvw1.x*pt3d_uvw1.x + pt3d_uvw1.y*pt3d_uvw1.y + 3*pt3d_uvw1.z*pt3d_uvw1.z;

      const float weight = 1.0 - dist_2;
      if ( weight <= 0.0 )
      {
        continue;
      }

      float dx = 0.5 * weight * ( patch[v][u+1] + patch[v+1][u+1] - patch[v][u] - patch[v+1][u] );
      float dy = 0.5 * weight * ( patch[v+1][u] + patch[v+1][u+1] - patch[v][u] - patch[v][u+1] );

      if ( !isnan(dx) && !isnan(dy) )
      {
        PtInfo ptInfo;

        // get pixel intensity
        ptInfo.intensity = 1;
        //ptInfo.grad.x = interpolateKernel<iiDx>( ii, x, y, s );
        //ptInfo.grad.y = interpolateKernel<iiDy>( ii, x, y, s );

        // correct for scaling along axis
        ptInfo.grad = Point2f(dx,dy);

        // the weight will be
        // - lower further away from the center
        // - higher for higher depth (one pixel covers more surface there, and the sampling is sparser)
        ptInfo.weight = weight;
        ptInfo.pos = pt3d_uvw1;

        float val = smoothed_img[int(pixel.y)][int(pixel.x)];
        ptInfo.intensity = val;

        pt_infos.push_back( ptInfo );

        // save gradient info for orientation histogram
        if ( finite(ptInfo.grad.x) && finite(ptInfo.grad.y) )
        {
          PtGradient grad;
          grad.grad_mag = sqrt ( ptInfo.grad.x*ptInfo.grad.x + ptInfo.grad.y*ptInfo.grad.y ) * ptInfo.weight;
          if ( grad.grad_mag > 0 )
          {
            grad.grad_ori = atan2 ( ptInfo.grad.y, ptInfo.grad.x );
            gradients.push_back( grad );
          }
        }

#ifdef DESC_DEBUG_IMG
        Size2f bsize( float(PATCH_MUL)*weight,float(PATCH_MUL)*weight );
        cv::RotatedRect box(uv_reproj*PATCH_MUL, bsize, angle/M_PI*180.0 );
        ellipse( points3d_img, box, cv::Scalar(val,0,0),-1, CV_AA );
        ellipse( points3d_img, box, cv::Scalar(0.5,0,0),1, CV_AA );

        patch_weights[v][u] = weight;
        patch_dx[v][u] = dx;
        patch_dy[v][u] = dy;

        if ( !isnan( patch_dx[v][u] ) && !isnan( patch_dy[v][u] ) )
        {
          Size2f bsize( 3,3 );
          cv::RotatedRect box(pixel, bsize, angle/M_PI*180.0 );
          ellipse( display_img, box, cv::Scalar(0,255,0), 1, CV_AA );
        }
#endif
      }
    }
  }
#endif
