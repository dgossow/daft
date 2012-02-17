/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <rgbd_features_cv/keypoint3d.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <math.h>

const int draw_shift_bits = 4;
const int draw_multiplier = 1 << draw_shift_bits;

namespace cv
{

static inline void _drawKeypoint3D( Mat& img, const KeyPoint3D& p, const Scalar& color, int flags )
{
  CV_Assert( !img.empty() );


  if( flags & DrawMatchesFlags::DRAW_RICH_KEYPOINTS )
  {
    float angle = atan2( (double)p.affine_mat(0,1), (double)p.affine_mat(0,0) );
    Size2f bsize( 2*cv::norm( p.affine_mat.row(0) ), 2*cv::norm( p.affine_mat.row(1) ) );

    cv::RotatedRect box( p.pt, bsize, angle/M_PI*180.0 );

    ellipse( img, box, color, 1, 16 );
  }
  else
  {
      // draw center with R=3
      int radius = 3 * draw_multiplier;
      Point center( cvRound(p.pt.x * draw_multiplier), cvRound(p.pt.y * draw_multiplier) );
      circle( img, center, radius, color, 1, CV_AA, draw_shift_bits );
  }
}



void drawKeypoints3D( const Mat& image, const vector<KeyPoint3D>& keypoints, Mat& outImage,
    const Scalar& _color, int flags )
{
  if( !(flags & DrawMatchesFlags::DRAW_OVER_OUTIMG) )
  {
      if( image.type() == CV_8UC3 )
      {
          image.copyTo( outImage );
      }
      else if( image.type() == CV_8UC1 )
      {
          cvtColor( image, outImage, CV_GRAY2BGR );
      }
      else
      {
          CV_Error( CV_StsBadArg, "Incorrect type of input image.\n" );
      }
  }

  RNG& rng=theRNG();
  bool isRandColor = _color == Scalar::all(-1);

  CV_Assert( !outImage.empty() );

  vector<KeyPoint3D>::const_iterator it = keypoints.begin(),
                                   end = keypoints.end();
  for( ; it != end; ++it )
  {
      Scalar color = isRandColor ? Scalar(rng(256), rng(256), rng(256)) : _color;
      _drawKeypoint3D( outImage, *it, color, flags );
  }
}

// convert 3d-keypoints into regular ones
std::vector<KeyPoint> makeKeyPoints( vector<KeyPoint3D> kp )
{
  std::vector<KeyPoint> kp_out;
  kp_out.reserve( kp.size() );

  for( std::vector<KeyPoint3D>::iterator it=kp.begin(); it!=kp.end(); ++it )
  {
    kp_out.push_back( *it );
  }
  return kp_out;
}

}
