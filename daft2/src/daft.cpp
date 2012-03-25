/*
 * RGBD Features -> OpenCV bridge
 * Copyright (C) 2011 David Gossow
 */

#include "daft.h"
#include "filter_kernels.h"
#include "feature_detection.h"
#include "descriptor.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/timer.hpp>

#include <cmath>

namespace cv {
namespace daft2 {

//#define SHOW_DEBUG_WIN
//#define FIND_MAXKP

DAFT::DAFT(const DetectorParams & detector_params) :
    params_(detector_params) {

}

DAFT::~DAFT() {

}

void DAFT::detect(const cv::Mat &image, const cv::Mat &depth_map_orig,
    cv::Matx33f K, std::vector<KeyPoint3D> & kp ) {
  if (image.size != depth_map_orig.size) {
    return;
  }

  Mat gray_image_orig = image;

  // Convert RGB to Grey image
  if (image.type() == CV_8UC3) {
    cvtColor(image, gray_image_orig, CV_BGR2GRAY);
  }

  // Construct integral image
  Mat1d ii;

  switch (gray_image_orig.type())
  {
  case CV_8U:
  {
    cv::Mat1b m_in = gray_image_orig;
    integral2( m_in, ii, 1.0/255.0 );
  }
  break;
  case CV_64F:
  {
    cv::Mat1d m_in = gray_image_orig;
    integral2( m_in, ii );
  }
  break;
  case CV_32F:
  {
    cv::Mat1f m_in = gray_image_orig;
    integral2( m_in, ii );
  }
  break;
  default:
    //return;
    break;
  }

  // Convert depth map to floating point
  cv::Mat1f depth_map;
  if (depth_map_orig.type() == CV_16U) {
    depth_map_orig.convertTo(depth_map, CV_32F, 0.001, 0.0);
  } else {
    depth_map = depth_map_orig;
  }

  // Initialize parameters

  double base_scale = params_.base_scale_;
  int n_octaves = params_.scale_levels_;

  int max_px_scale =
      params_.max_px_scale_ == params_.AUTO ?
          std::min(image.rows, image.cols) / 10 : params_.max_px_scale_;

  const float f = K(0, 0);

  // Integrate Depth + Depth count (#of valid pixels)
  cv::Mat1d ii_depth_map(depth_map.rows + 1, depth_map.cols + 1);
  cv::Mat_<uint64_t> ii_depth_count(depth_map.rows + 1, depth_map.cols + 1);

  for (int y = 0; y < depth_map.rows + 1; y++) {
    double row_sum = 0;
    double row_count = 0;
    for (int x = 0; x < depth_map.cols + 1; x++) {
      if (x == 0 || y == 0) {
        ii_depth_map[y][x] = 0;
        ii_depth_count[y][x] = 0;
      } else {
        float depth = depth_map[y - 1][x - 1];
        if (!isnan(depth)) {
          row_sum += depth;
          row_count += 1;
        }
        ii_depth_map[y][x] = row_sum + ii_depth_map[y - 1][x];
        ii_depth_count[y][x] = row_count + ii_depth_count[y - 1][x];
      }
    }
  }

  // Compute scale map from depth map
  Mat1f scale_map(gray_image_orig.rows, gray_image_orig.cols);
  Mat1f::iterator scale_it = scale_map.begin();
  Mat1f::iterator scale_map_end = scale_map.end();
  Mat1f::iterator depth_it = depth_map.begin();

  if (params_.scale_levels_ == params_.AUTO)
  {
    float min_scale_fac = std::numeric_limits<float>::infinity();
    float max_scale_fac = 0;

    for (; scale_it != scale_map_end; ++scale_it, ++depth_it) {
      if (finite(*depth_it)) {
        float s = f / *depth_it;
        *scale_it = s;
        if (s > max_scale_fac)
          max_scale_fac = s;
        if (s < min_scale_fac)
          min_scale_fac = s;
      } else {
        *scale_it = -1.0f;
      }
    }

    if (!finite(min_scale_fac) || max_scale_fac == 0) {
      return;
    }

    double delta_n_min = params_.min_px_scale_
        / (max_scale_fac * params_.base_scale_);
    double delta_n_max = max_px_scale / (min_scale_fac * params_.base_scale_);

    double n_min = std::ceil(log(delta_n_min) / log(params_.scale_step_));
    double n_max = std::floor(log(delta_n_max) / log(params_.scale_step_));

    base_scale = params_.base_scale_
        * std::pow((double) params_.scale_step_, n_min);
    n_octaves = n_max - n_min + 1;
  }
  else
  {
    for (; scale_it != scale_map_end; ++scale_it, ++depth_it) {
      if (isnan(*depth_it)) {
        *scale_it = -1.0f;
      } else {
        *scale_it = f / *depth_it;
      }
    }
  }

  kp.clear();
  kp.reserve(50000);

#if 0
  cv::Mat1f grad_map_x(depth_map.size());
  cv::Mat1f grad_map_y(depth_map.size());
  convolveAffine<gradX>(ii, scale_map, ii_depth_map, ii_depth_count, base_scale,
      params_.min_px_scale_, max_px_scale, grad_map_x);
  convolveAffine<gradY>(ii, scale_map, ii_depth_map, ii_depth_count, base_scale,
      params_.min_px_scale_, max_px_scale, grad_map_y);
  cv::imshow("grad_map_x", grad_map_x * 0.2 + 0.5);
  cv::imshow("grad_map_y", grad_map_y * 0.2 + 0.5);
#endif

  //cv::imshow( "img", gray_image_orig );

  std::vector<Mat1f> smoothed_imgs(n_octaves+1);
  std::vector<Mat2f> depth_grads(n_octaves+1);

  // TODO: compute depth gradient

  // compute depth-normalized image pyramid
  double scale = base_scale;
  for (int octave = 0; octave < n_octaves+1; octave++, scale *= params_.scale_step_)
  {
    std::cout << "l " << octave << std::endl;

    Mat1f& smoothed_img = smoothed_imgs[octave];
    Mat2f& depth_grad = depth_grads[octave];

    // compute filter response for all pixels
    switch (params_.det_type_) {
    case DetectorParams::DET_DOB:
      if (params_.affine_) {
        convolveAffine<boxAffine>(ii, scale_map, ii_depth_map, ii_depth_count,
            scale, params_.min_px_scale_, max_px_scale, smoothed_img, depth_grad );
      } else {
        convolve<box>(ii, scale_map, scale, params_.min_px_scale_,
            max_px_scale, smoothed_img);
      }
      break;
    case DetectorParams::DET_LAPLACE:
      if (params_.affine_) {
          convolveAffine<gaussAffine>(ii, scale_map, ii_depth_map, ii_depth_count,
              scale, params_.min_px_scale_, max_px_scale, smoothed_img, depth_grad );
      } else {
        convolve<gauss>(ii, scale_map, scale, params_.min_px_scale_,
            max_px_scale, smoothed_img);

        //showBig( 128, sGaussKernel.asCvImage() + 0.5f, "gauss" );
        //std::cout << "cv::sum(sGaussKernel.asCvImage()) " << cv::sum(sGaussKernel.asCvImage())[0] << std::endl;
      }
      break;
    default:
      return;
    }

#ifdef SHOW_DEBUG_WIN
    {
      static int i=0;
      cv::Mat display_image;
      smoothed_img.convertTo( display_image, CV_8UC1, 255, 0.0 );
      std::ostringstream s;
      s << "frame # " << i;
      //cv::putText( display_image, s.str( ), Point(10,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0) );

      s.str("");
      s << "Smooth Detector type=" << params_.det_type_ << " max=" << params_.max_search_algo_ << " scale = " << scale << " affine = " << params_.affine_;
      imshow( s.str(), display_image );

      /*
       cv::imshow( "dep orig", depth_map_orig );
       cv::imshow( "dep", depth_map );
       cv::imshow( "det", detector_image );
       cv::imshow( "scale", scale_map );
       */

      i++;
    }
#endif
  }

  Mat1f response_map;
  response_map.create(gray_image_orig.rows, gray_image_orig.cols);

  // compute difference of gaussians and detect extrema
  scale = base_scale;
  for (int octave = 0; octave < n_octaves; octave++, scale *= params_.scale_step_)
  {
    diff( smoothed_imgs[octave+1], smoothed_imgs[octave], response_map );
    Mat2f& depth_grad = depth_grads[octave];

    // save index where new kps will be inserted
    unsigned kp_first = kp.size();

    // find maxima in response
    switch (params_.max_search_algo_) {
    case DetectorParams::MAX_WINDOW:
      if ( params_.affine_ ) {
        findMaximaAffine(response_map, scale_map, depth_grad,
            scale, params_.det_threshold_, kp);
      }
      else {
        findMaxima(response_map, scale_map, scale, params_.det_threshold_, kp);
      }
      break;
    case DetectorParams::MAX_FAST:
      findMaximaMipMap(response_map, scale_map, scale, params_.det_threshold_,
          kp);
      break;
    case DetectorParams::MAX_EVAL: {
      boost::timer timer;
      timer.restart();
      for (int i = 0; i < 100; i++) {
        kp.clear();
        kp.reserve(50000);
        findMaximaMipMap(response_map, scale_map, scale, params_.det_threshold_,
            kp);
      }
      std::cout << "findMaximaMipMap execution time [ms]: "
          << timer.elapsed() * 10 << std::endl;
      timer.restart();
      for (int i = 0; i < 100; i++) {
        kp.clear();
        kp.reserve(50000);
        findMaxima(response_map, scale_map, scale, params_.det_threshold_, kp);
      }
      std::cout << "findMaxima execution time [ms]: " << timer.elapsed() * 10
          << std::endl;
    }
      break;
    default:
      return;
    }

    std::cout << "l " << octave << ": " << kp.size()-kp_first << " keypoints found." << std::endl;

    // assign octave
    for ( unsigned k=kp_first; k<kp.size(); k++ )
    {
      kp[k].octave = octave;
    }

#ifdef SHOW_DEBUG_WIN
    {
      static int i=0;
      cv::Mat display_image;
      response_map.convertTo( display_image, CV_8UC1, 900, 0.0 );

      vector<KeyPoint3D> kp2;
      for ( unsigned k=kp_first; k<kp.size(); k++ )
      {
        KeyPoint3D kp_curr = kp[k];
        int kp_x = kp[k].pt.x;
        int kp_y = kp[k].pt.y;

        Vec2f grad;
        if ( computeGradient( ii_depth_map, ii_depth_count, kp_x, kp_x, kp[k].size*0.25f, grad ) )
        {
          getAffine( grad, kp_x, kp_y, kp[k].size*0.25f, kp[k].world_size*0.25f,
              kp[k].affine_angle, kp[k].affine_major, kp[k].affine_minor,
              kp[k].normal );
          // keypoint shall cover outer and inner and wants size not radius
          kp[k].affine_minor *= 4.0f;
          kp[k].affine_major *= 4.0f;
          kp2.push_back(kp_curr);
        }
      }

//      cv::drawKeypoints3D( display_image, kp2, display_image, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

      std::ostringstream s;
      s << "frame # " << i;
      cv::putText( display_image, s.str( ), Point(10,40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,255,0) );

      s.str("");
      s << "Response Detector type=" << params_.det_type_ << " max=" << params_.max_search_algo_ << " scale = " << scale << " affine = " << params_.affine_;
      imshow( s.str(), display_image );

      /*
       cv::imshow( "dep orig", depth_map_orig );
       cv::imshow( "dep", depth_map );
       cv::imshow( "det", detector_image );
       cv::imshow( "scale", scale_map );
       */

      i++;
    }
#endif
  }

  // filter found maxima by applying a threshold on a second kernel
  switch (params_.pf_type_) {
  case DetectorParams::PF_NONE:
    break;
  case DetectorParams::PF_NEIGHBOURS:
    filterKpNeighbours(response_map, params_.pf_threshold_, kp);
    break;
  case DetectorParams::PF_PRINC_CURV_RATIO: {
    float r = params_.pf_threshold_;
    float r_thresh = (r + 1) * (r + 1) / r;
    if (params_.affine_) {
      filterKpKernelAffine<princCurvRatioAffine>(ii, r_thresh, kp);
    } else {
      filterKpKernel<princCurvRatio>(ii, r_thresh, kp);
#if 0
      showBig( 128, 3.0f*sDxxKernel.asCvImage() + 0.5f, "dxx" );
      showBig( 128, 3.0f*sDyyKernel.asCvImage() + 0.5f, "dyy" );
      showBig( 128, 3.0f*sDxyKernel.asCvImage() + 0.5f, "dxy" );
#endif
    }
    break;
  }

  default:
    return;
  }

  // assign 3d points, normals and local affine params
  float f_inv = 1.0 / f;
  float cx = K(0, 2);
  float cy = K(1, 2);
  vector<KeyPoint3D> kp2;
  kp2.reserve(kp.size());
  for (unsigned k = 0; k < kp.size(); k++) {
    int kp_x = kp[k].pt.x;
    int kp_y = kp[k].pt.y;
    getPt3d(f_inv, cx, cy, kp_x, kp_y, depth_map[kp_y][kp_x], kp[k].pt3d);

    Vec2f depth_grad;
    computeGradient( ii_depth_map, ii_depth_count, kp_x, kp_y, kp[k].size * 0.25f, depth_grad );

    if (getAffine(depth_grad, kp_x, kp_y, kp[k].size * 0.25f,
        kp[k].world_size * 0.25f, kp[k].affine_angle, kp[k].affine_major,
        kp[k].affine_minor, kp[k].normal))
    {
      // keypoint shall cover outer and inner and wants size not radius
      kp[k].affine_minor *= 4.0f;
      kp[k].affine_major *= 4.0f;

      Mat1f& smoothed_img = smoothed_imgs[kp[k].octave];
      Mat1f& smoothed_img2 = smoothed_imgs[kp[k].octave+1];
      getDesc<10,20>( smoothed_img, smoothed_img2, kp[k], depth_map, K );

      kp2.push_back(kp[k]);
    }
  }
  kp = kp2;

#ifdef FIND_MAXKP
  cv::Mat1f patch1, patch2;

  cv::Mat display_image;
  cv::drawKeypoints3D( image, kp, display_image, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

  // draw patch for strongest keypoint
  float resp_max=0;
  unsigned k_max=0;
  for ( unsigned k=0; k<kp.size(); k++ )
  {
    if ( kp[k].response > resp_max )
    {
      k_max = k;
      resp_max = kp[k].response;
    }
  }

  if ( kp.size() > 0 )
  {
    vector<KeyPoint3D> kp2;
    kp2.push_back(kp[k_max]);
    cv::Mat tmp;
    response_map.convertTo( tmp, CV_8UC1, 900, 0.0 );
    cv::drawKeypoints3D( tmp, kp2, tmp, cv::Scalar(0,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
    imshow("max_kp",tmp);

    int k=k_max;
    std::stringstream s;
    s << "max response patch";

    const float scale_fac = 3; //40.0/2.0;

    Mat1f& smoothed_img = smoothed_imgs[kp[k].octave];
    Mat1f& smoothed_img2 = smoothed_imgs[kp[k].octave+1];
    Mat2f& depth_grad = depth_grads[kp[k].octave];

    getDesc<10,20>( smoothed_img, smoothed_img2, kp[k], depth_map, K );
  }

  imshow( "rgb", display_image );
#endif

}

}
}
