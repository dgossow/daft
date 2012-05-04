/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_DAFT2_H__
#define __DAFT2_DAFT2_H__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features3d/features3d.hpp>

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
    enum { DET_FELINE=0 };
    enum { PF_NONE=0, PF_NEIGHBOURS=2, PF_PRINC_CURV_RATIO=3 };
    enum { MAX_WINDOW=0, MAX_FAST=2 };

    enum { AUTO=-1 };

    /** default constructor */
    DetectorParams(
        float base_scale = 1,
        float scale_factor = 2.0,
        int scale_levels = AUTO,
        float min_px_scale = 2.5,
        float max_px_scale = AUTO,
        int detector_type = DET_FELINE,
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

    /** The smallest scale (in meters) at which to search for features */
    double base_scale_;

    /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
    float scale_step_;

    /** The number of levels in the scale pyramid */
    int scale_levels_;

    float min_px_scale_;
    float max_px_scale_;

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
    DescriptorParams(
        int patch_size=20,
        int octave_offset=0 ) :
          patch_size_(patch_size),
          octave_offset_(octave_offset)
    {
    }

    /** For the descriptor computation, take patch_size*patch_size samples */
    int patch_size_;

    /** Sampling for the descriptor is done on the scale level 2^octave_offset_ * keypoint.world_size */
    int octave_offset_;
  };


  /** Constructor
   * @param detector_params parameters for the detector
   * @param desc_params parameters for the descriptor
   */
  DAFT(const DetectorParams & detector_params = DetectorParams(),
      const DescriptorParams & desc_params=DescriptorParams() );

  ~DAFT();

  /** Detect salient keypoints using a pair of depth and intensity images
   * @param image     the image to compute the features and descriptors on
   * @param depth_map the depth image in meters (float,double) or millimeters (int16)
   * @param K         matrix with intrinsic camera parameters
   * @param keypoints The resulting keypoints
   * @param desc      float matrix with row-wise descriptors
   */
  void operator()(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints, cv::Mat1f& desc );
  void operator()(const cv::Mat &image, const cv::Mat1b &mask, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints, cv::Mat1f& desc );

private:

  bool prepareData(const cv::Mat &image, const cv::Mat &depth_map_orig,
      Mat& gray_image, Mat1d& ii, cv::Mat1f& depth_map );

  /** Parameters tuning RgbdFeatures */
  DetectorParams det_params_;
  DescriptorParams desc_params_;
};

}
}

#endif
