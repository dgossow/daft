/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <opencv2/features2d/features2d.hpp>

#ifndef __DAFT_H__
#define __DAFT_H__

#include "keypoint3d.h"

namespace cv
{

/*!
  Extracts keypoints from an image based on intensity and depth data
*/
class DAFT2
{
public:

  struct DetectorParams
  {
    enum { DET_DOB=0, DET_LAPLACE=1 };
    enum { PF_NONE=0, PF_HARRIS=1, PF_NEIGHBOURS=2, PF_PRINC_CURV_RATIO=3 };
    enum { MAX_WINDOW=0, MAX_WINDOW_AFFINE=1, MAX_FAST=2, MAX_EVAL=3 };

    enum { AUTO=-1 };

    static const int DEFAULT_SCALE_LEVELS = AUTO;
    static const float DEFAULT_BASE_SCALE = 1;
    static const float DEFAULT_SCALE_FACTOR = 2.0;
    static const int DEFAULT_MIN_PX_SCALE = 2;
    static const int DEFAULT_MAX_PX_SCALE = AUTO;
    static const int DEFAULT_DET = DET_DOB;
    static const float DEFAULT_DET_THRESHOLD = 0.02;
    static const int DEFAULT_PF_TYPE = PF_PRINC_CURV_RATIO;
    static const float DEFAULT_PF_THRESHOLD = 10.0;
    static const int DEFAULT_MAX_ALGO = MAX_FAST;
    static const bool DEFAULT_AFFINE = true;

    /** default constructor */
    DetectorParams(
        float base_scale = DEFAULT_BASE_SCALE,
        float scale_factor = DEFAULT_SCALE_FACTOR,
        int scale_levels = DEFAULT_SCALE_LEVELS,
        int min_px_scale = DEFAULT_MIN_PX_SCALE,
        int max_px_scale = DEFAULT_MAX_PX_SCALE,
        int detector_type = DEFAULT_DET,
        float det_threshold = DEFAULT_DET_THRESHOLD,
        int postfilter_type = DEFAULT_PF_TYPE,
        float pf_threshold = DEFAULT_PF_THRESHOLD,
        int max_search_algo = DEFAULT_MAX_ALGO,
        bool affine = DEFAULT_AFFINE):
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

  /** Constructor
   * @param detector_params parameters to use
   */
  DAFT2(const DetectorParams & detector_params = DetectorParams());

  ~DAFT2();

  /** Detect salient keypoints on a rectified depth+intensity image
   * @param image the image to compute the features and descriptors on
   * @param depthImage the depth image (depth values in meters)
   * @param cameraMatrix
   * @param keypoints the resulting keypoints
   */
  void detect(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f camera_matrix,
      std::vector<KeyPoint3D> & keypoints);

private:

  /** Parameters tuning RgbdFeatures */
  DetectorParams params_;
};

}

#endif
