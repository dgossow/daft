/*
 * extract_keypoints.h
 *
 *  Created on: May 4, 2012
 *      Author: gossow
 */

#ifndef EXTRACT_KEYPOINTS_H_
#define EXTRACT_KEYPOINTS_H_

#include <boost/function.hpp>

#include <opencv2/opencv.hpp>

#include <daft2/daft.h>

#include "external/sift/Sift.h"
#include "external/parallelsurf/KeyPointDetector.h"
#include "external/parallelsurf/KeyPointDescriptor.h"
#include "external/parallelsurf/Image.h"

#define daft_ns cv::daft2

typedef boost::function< void (
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors ) > GetKpFunc;

void getDaftKp(
    daft_ns::DAFT::DetectorParams p_det,
    daft_ns::DAFT::DescriptorParams p_desc,
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors );

void getSurfKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors );

void getSiftKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors );

#ifdef USE_ORB
void getOrbKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors );
#endif

#endif /* EXTRACT_KEYPOINTS_H_ */
