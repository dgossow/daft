/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#include <opencv2/features2d/features2d.hpp>

#ifndef __RGBD_CV_H__
#define __RGBD_CV_H__

namespace cv
{
/*
void make3DKeypoints( const cv::Matx33f& camera_matrix,
    const cv::Mat1d& scale_map,
    std::vector< KeyPoint >& kp_in,
    std::vector< KeyPoint >& kp_in
    );
*/

/*!
  RGBD features implementation.
*/
class RgbdFeatures
{
public:

  struct DetectorParams
  {
    enum { DET_DOB=0, DET_LAPLACE=1, DET_HARRIS=2 };
    enum { PF_NONE=0, PF_HARRIS=1 };
    enum { MAX_WINDOW=0, MAX_FAST=1, MAX_EVAL=3 };

    static const unsigned int DEFAULT_N_LEVELS = 1;
    static const float DEFAULT_DET_THRESHOLD = 0.02;
    static const float DEFAULT_PF_THRESHOLD = 0.0;
    static const float DEFAULT_BASE_SCALE = 0.025;
    static const float DEFAULT_SCALE_FACTOR = 2.0;
    static const float DEFAULT_MAX_ALGO = MAX_WINDOW;

    /** default constructor */
    DetectorParams(
        float base_scale = DEFAULT_BASE_SCALE,
        float scale_factor = DEFAULT_SCALE_FACTOR,
        unsigned int n_levels = DEFAULT_N_LEVELS,
        int detector_type = DET_DOB,
        float det_threshold = DEFAULT_DET_THRESHOLD,
        int postfilter_type = PF_HARRIS,
        float pf_threshold = DEFAULT_PF_THRESHOLD,
        int max_search_algo = DEFAULT_MAX_ALGO ):
          base_scale_(base_scale),
          scale_step_(scale_factor),
          scale_levels_(n_levels),
          det_type_(detector_type),
          det_threshold_(det_threshold),
          pf_type_(postfilter_type),
          pf_threshold_(pf_threshold),
          max_search_algo_(max_search_algo)
    {
    }

    /** The smallest scale (in meters) at which to search for features */
    double base_scale_;

    /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
    float scale_step_;

    /** The number of levels in the scale pyramid */
    unsigned int scale_levels_;

    /** Which detector to use (Hessian, Difference-of-Boxes, Harris) */
    int det_type_;

    /** Minimal response threshold for the detector */
    double det_threshold_;

    /** Postfilter applied to output of first detector (None, Hessian) */
    int pf_type_;

    /** Minimal response threshold for the post filter */
    double pf_threshold_;

    /** Postfilter applied to output of first detector (None, Hessian) */
    int max_search_algo_;

  };

  struct Keypoint: public cv::KeyPoint
  {
    CV_PROP_RW float physical_size; //!< diameter (in meters) of the meaningful keypoint neighborhood
    CV_PROP_RW Point3f pt3d; //!< 3D position in the camera frame
  };

  /** Constructor
   * @param detector_params parameters to use
   */
  RgbdFeatures(const DetectorParams & detector_params = DetectorParams());

  ~RgbdFeatures();

  /** Detect salient keypoints on a rectified depth+intensity image
   * @param image the image to compute the features and descriptors on
   * @param depthImage the depth image (depth values in meters)
   * @param cameraMatrix
   * @param keypoints the resulting keypoints
   */
  void detect(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f camera_matrix,
      std::vector<cv::KeyPoint> & keypoints);

private:

  /** Parameters tuning RgbdFeatures */
  DetectorParams detector_params_;
};

}

#endif
