/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
*/

#ifndef __DAFT2_DAFT2_H__
#define __DAFT2_DAFT2_H__

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features3d/features3d.hpp>

#include <set>

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
    enum { MAX_WINDOW=0, MAX_FAST=2 };

    enum { AUTO=-1 };

    /** default constructor */
    DetectorParams(
        bool affine_multiscale = false,
        float base_scale = 1,
        float scale_step = 2.0,
        int scale_levels = AUTO,
        float min_px_scale = 2.5,
        float max_px_scale = AUTO,
        float min_dist = 2.0,
        int detector_type = DET_FELINE,
        float det_threshold = 0.02,
        double max_princ_curv_ratio = 10.0,
        int max_search_algo = MAX_WINDOW,
        bool affine = true,
        unsigned max_num_kp = std::numeric_limits<unsigned>::max() ):
          affine_multiscale_(affine_multiscale),
          base_scale_(base_scale),
          scale_step_(scale_step),
          scale_levels_(scale_levels),
          min_px_scale_(min_px_scale),
          max_px_scale_(max_px_scale),
          min_dist_(min_dist),
          det_type_(detector_type),
          det_threshold_(det_threshold),
          max_princ_curv_ratio_(max_princ_curv_ratio),
          max_search_algo_(max_search_algo),
          affine_(affine),
          max_num_kp_(max_num_kp)
    {
    }

    /** If true, compute affine parameters for each scale separately (slower, but maybe more correct) */
    bool affine_multiscale_;

    /** The smallest scale (in meters) at which to search for features */
    double base_scale_;

    /** Coefficient by which we divide the dimensions from one scale pyramid level to the next */
    float scale_step_;

    /** The number of levels in the scale pyramid */
    int scale_levels_;

    float min_px_scale_;
    float max_px_scale_;

    /** Minimal distance between two keypoints on the same scale */
    float min_dist_;

    /** Which detector to use */
    int det_type_;

    /** Minimal response threshold for the detector */
    double det_threshold_;

    /** Max. principal curvature ratio ( no check if < 1.0) */
    double max_princ_curv_ratio_;

    /** How to search for maxima? */
    int max_search_algo_;

    /** Use local affine transformation for detection */
    bool affine_;

    /** If > 0, limit the number of keypoints */
    unsigned max_num_kp_;
  };


  struct DescriptorParams
  {
    /** default constructor */
    DescriptorParams(
        int patch_size=20,
        int octave_offset=0,
        float z_thickness=0.3 ) :
          patch_size_(patch_size),
          octave_offset_(octave_offset),
          z_thickness_(z_thickness)
    {
    }

    /** For the descriptor computation, take patch_size*patch_size samples */
    int patch_size_;

    /** Sampling for the descriptor is done on the scale level 2^octave_offset_ * keypoint.world_size */
    int octave_offset_;

    /** defines the thickness of the ellipsiod from where points are considered (1.0 for a sphere) */
    float z_thickness_;
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
  void operator()(const cv::Mat &image, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints );

  /** Detect salient keypoints using a pair of depth and intensity images
   * @param image     the image to compute the features and descriptors on
   * @param mask      mask for keypoint filtering (0=reject)
   * @param depth_map the depth image in meters (float,double) or millimeters (int16)
   * @param K         matrix with intrinsic camera parameters
   * @param keypoints The resulting keypoints
   * @param desc      float matrix with row-wise descriptors
   */
  void operator()(const cv::Mat &image, const cv::Mat1b &mask, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints, cv::Mat1f& desc );
  void operator()(const cv::Mat &image, const cv::Mat1b &mask, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints );

private:

  void computeScaleMap( const Mat1f &depth_map, const Mat1b &mask, float f, Mat1f &scale_map );

  void getOctaves( const Mat1f &scale_map, float max_px_scale, std::set<int> &pyr_octaves, std::set<int> &det_octaves );

  void computeImpl(const cv::Mat &image, const cv::Mat1b &mask, const cv::Mat &depth_map, cv::Matx33f K,
      std::vector<cv::KeyPoint3D> & keypoints, cv::Mat1f& desc, bool computeDescriptors );

  bool prepareData(const cv::Mat &image, const cv::Mat &depth_map_orig,
      Mat& gray_image, Mat1d& ii, cv::Mat1f& depth_map, cv::Mat1b& mask );

  void computeAffineMaps(
      std::set<int>& octaves,
      cv::Mat1f& depth_map,
      cv::Mat1f& scale_map,
      float f,
      std::map< int, Mat1f>& smoothed_depth_maps,
      std::map< int, Mat3f >& affine_maps );

  /** Parameters tuning RgbdFeatures */
  DetectorParams det_params_;
  DescriptorParams desc_params_;

public:
  std::map< int, Mat1f > smoothed_imgs;
  std::map< int, Mat1f > response_maps;
};

}
}

#endif
