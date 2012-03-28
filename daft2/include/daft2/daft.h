/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <opencv2/features2d/features2d.hpp>

#ifndef __DAFT2_DAFT2_H__
#define __DAFT2_DAFT2_H__

#include "features3d/keypoint3d.h"

namespace cv
{
namespace daft2
{

/*!
  Extracts keypoints from an image based on intensity and depth data
*/
class DAFT
{
public:

  struct DetectorParams
  {
    enum { DET_BOX=0, DET_9X9=1, DET_FELINE=2 };
    enum { PF_NONE=0, PF_NEIGHBOURS=2, PF_PRINC_CURV_RATIO=3 };
    enum { MAX_WINDOW=0, MAX_FAST=2, MAX_EVAL=3 };

    enum { AUTO=-1 };

    /** default constructor */
    DetectorParams(
        float base_scale = AUTO,
        float scale_factor = 2.0,
        int scale_levels = AUTO,
        int min_px_scale = 3,
        int max_px_scale = AUTO,
        int detector_type = DET_BOX,
        float det_threshold = 0.02,
        int postfilter_type = PF_PRINC_CURV_RATIO,
        float pf_threshold = 5.0,
        int max_search_algo = MAX_WINDOW,
        bool affine = true):
          base_scale_(base_scale),
          scale_step_(scale_factor),
          scale_levels_(scale_levels),
          min_px_scale_(min_px_scale),
          max_px_scale_(max_px_scale),
          det_type_(detector_type),
          det_threshold_(det_threshold),
          pf_type_(postfilter_type),
          pf_threshold_(pf_threshold),
          max_search_algo_(max_search_algo),
          affine_(affine)
    {
    }


    struct DescriptorParams
    {
      /** default constructor */
      DescriptorParams( int octave_offset=0 ) : octave_offset_(octave_offset)
      {
      }

      int octave_offset_;
    };

    /** The smallest scale (in meters) at which to search for features */
    double base_scale_;

    /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
    float scale_step_;

    /** The number of levels in the scale pyramid */
    int scale_levels_;

    int min_px_scale_;
    int max_px_scale_;

    /** Which detector to use */
    int det_type_;

    /** Minimal response threshold for the detector */
    double det_threshold_;

    /** Postfilter applied to output of first detector */
    int pf_type_;

    /** Minimal response threshold for the post filter */
    double pf_threshold_;

    /** How to search for maxima? */
    int max_search_algo_;

    /** Use local affine transformation for detection */
    bool affine_;
  };


  struct DescriptorParams
  {
    /** default constructor */
    DescriptorParams( int octave_offset=0 ) : octave_offset_(octave_offset)
    {
    }

    int octave_offset_;
  };


  /** Constructor
   * @param detector_params parameters to use
   */
  DAFT(const DetectorParams & detector_params = DetectorParams(), const DescriptorParams & desc_params=DescriptorParams() );

  ~DAFT();

  /** Detect salient keypoints on a rectified depth+intensity image
   * @param image the image to compute the features and descriptors on
   * @param depthImage the depth image (depth values in meters)
   * @param cameraMatrix
   * @param keypoints the resulting keypoints
   */
  void detect(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f camera_matrix,
      std::vector<cv::KeyPoint3D> & keypoints );

private:

  void prepareData(const cv::Mat &image, const cv::Mat &depth_map_orig,
      Mat& gray_image, Mat1d& ii, cv::Mat1f& depth_map );

  /** Parameters tuning RgbdFeatures */
  DetectorParams det_params_;
  DescriptorParams desc_params_;
};

}
}

#endif
