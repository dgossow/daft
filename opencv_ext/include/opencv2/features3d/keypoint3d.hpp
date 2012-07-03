/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#ifndef __CV_KEYPOINT3D_H__
#define __CV_KEYPOINT3D_H__

#include <opencv2/features2d/features2d.hpp>

#include <iostream>

namespace cv
{

struct KeyPoint3D: public cv::KeyPoint
{
  KeyPoint3D(
      float _x,
      float _y,
      float _world_size,
      float _affine_major,
      float _affine_minor,
      float _affine_angle,
      float _angle=-1,
      float _response=0,
      int _octave=0,
      int _class_id=-1)
      : KeyPoint( _x ,_y, sqrt(_affine_major*_affine_minor), _angle, _response, _octave, _class_id),
        world_size(_world_size), aff_major(_affine_major), aff_minor(_affine_minor), aff_angle(_affine_angle) {}

  KeyPoint3D( const KeyPoint& kp ): KeyPoint( kp ),
        world_size(0), aff_major(kp.size), aff_minor(kp.size), aff_angle(0) {}

  float world_size; //!< diameter (in meters) of the meaningful keypoint neighborhood

  // 3d transformation of keypoint frame
  CV_PROP_RW Point3f pt3d; //!< 3D position
  CV_PROP_RW Point3f normal; //!< 3D normal (TODO: Use quaternion)

  // affine approximation to a perspective projection of the tangent plane
  CV_PROP_RW float aff_major;
  CV_PROP_RW float aff_minor;
  CV_PROP_RW float aff_angle;
};

void drawKeypoints3D( const Mat& image, const vector<KeyPoint3D>& keypoints, Mat& outImage,
    const Scalar& color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT );

// convert 3d-keypoints into regular ones
vector<KeyPoint> makeKeyPoints( vector<KeyPoint3D> kp );

template<class KeyPointT >
bool compareResponse( const KeyPointT& kp1, const KeyPointT& kp2 )
{
  return kp1.response > kp2.response;
}

template<class KeyPointT>
vector<KeyPointT> getStrongest( size_t number, vector<KeyPointT> kp_in )
{
  std::sort( kp_in.begin( ), kp_in.end( ), compareResponse<KeyPointT> );
  if ( kp_in.size() > number )
  {
    kp_in.erase( kp_in.begin() + number, kp_in.end() );
  }
  //for (unsigned i=0; i<kp_in.size(); i++ ) std::cout << kp_in[i].response << std::endl;
  return kp_in;
}

}

#endif
