/*
 * extract_keypoints.cpp
 *
 *  Created on: May 4, 2012
 *      Author: gossow
 */


#include "rgbd_evaluator/extract_keypoints.h"


std::vector<cv::KeyPoint3D> makeKp3d(std::vector<cv::KeyPoint> kp) {
  std::vector<cv::KeyPoint3D> kp_3d;
  kp_3d.reserve(kp.size());
  for (size_t i = 0; i < kp.size(); i++) {
    kp_3d.push_back(kp[i]);
  }
  return kp_3d;
}

class VecIns: public parallelsurf::KeyPointInsertor {
public:
  VecIns(std::vector<parallelsurf::KeyPoint>& keyPoints) :
      m_KeyPoints(keyPoints) {
  }
  virtual ~VecIns() {};
  inline virtual void operator()(const parallelsurf::KeyPoint &keyPoint) {
    m_KeyPoints.push_back(keyPoint);
  }
private:
  std::vector<parallelsurf::KeyPoint>& m_KeyPoints;
};

void getSurfKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors )
{
  unsigned char** pixels = new unsigned char*[gray_img.rows];
  for (int y = 0; y < gray_img.rows; y++) {
    pixels[y] = new unsigned char[gray_img.cols];
    for (int x = 0; x < gray_img.cols; x++) {
      pixels[y][x] = gray_img.at<uchar>(y, x);
    }
  }
  parallelsurf::Image intImage((const unsigned char**) pixels, gray_img.cols,
      gray_img.rows);

  parallelsurf::KeyPointDetector detector;
  detector.setScoreThreshold(0.1 * t);

  parallelsurf::KeyPointDescriptor descriptor(intImage, false);
  std::vector<parallelsurf::KeyPoint> surf_kps;
  VecIns insertor(surf_kps);

  detector.detectKeyPoints(intImage, insertor);
  descriptor.assignOrientations(surf_kps.begin(), surf_kps.end());
  descriptor.makeDescriptors(surf_kps.begin(), surf_kps.end());

  if (surf_kps.size() == 0) {
    return;
  }

  int num_kp = surf_kps.size();

  std::vector<cv::KeyPoint> kps;

  int desc_len = surf_kps[0]._vec.size();
  descriptors.create(num_kp, desc_len);

  for (int i = 0; i < num_kp; i++) {
    cv::KeyPoint kp;
    parallelsurf::KeyPoint &surf_kp = surf_kps[i];

    if ( mask_img.rows == 0 || mask_img(surf_kp._y,surf_kp._x) != 0 )
    {
      kp.angle = surf_kp._ori;
      kp.class_id = 0;
      kp.octave = 0;
      kp.pt = cv::Point2f(surf_kp._x, surf_kp._y);
      kp.response = surf_kp._score;
      kp.size = surf_kp._scale * 7.768891354;
      kps.push_back(kp);

      for (int j = 0; j < desc_len; j++) {
        descriptors[kps.size()-1][j] = surf_kp._vec[j];
      }
    }
  }
  descriptors = descriptors( cv::Rect( 0, 0, desc_len, kps.size() ) ).clone();

  keypoints = makeKp3d(kps);
}

void getSiftKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors )
{
  Lowe::SIFT sift;
  sift.PeakThreshInit = 0.01 * t;

  Lowe::SIFT::KeyList lowe_kps;
  Lowe::SIFT::Image lowe_img(gray_img.rows, gray_img.cols);

  for (int y = 0; y < gray_img.rows; y++) {
    for (int x = 0; x < gray_img.cols; x++) {
      lowe_img.pixels[y][x] = gray_img.at<uchar>(y, x) / 255.0;
    }
  }

  lowe_kps = sift.getKeypoints(lowe_img);
  if (lowe_kps.size() == 0) {
    return;
  }

  int num_kp = lowe_kps.size();

  std::vector<cv::KeyPoint> kps;

  int desc_len = lowe_kps[0].ivec.size();
  descriptors.create(num_kp, desc_len);

  for (int i = 0; i < num_kp; i++) {
    Lowe::SIFT::Key& lowe_kp = lowe_kps[i];
    if ( mask_img.rows == 0 || mask_img(lowe_kp.row, lowe_kp.col) )
    {
      cv::KeyPoint kp;
      kp.angle = 0;
      kp.class_id = 0;
      kp.octave = 0;
      kp.pt = cv::Point2f(lowe_kp.col, lowe_kp.row);
      kp.response = lowe_kp.strength;
      kp.size = lowe_kp.scale * 6.14;
      kps.push_back(kp);
      for (int j = 0; j < desc_len; j++) {
        descriptors[kps.size()-1][j] = lowe_kp.ivec[j];
      }
    }
  }
  descriptors = descriptors( cv::Rect( 0, 0, desc_len, kps.size() ) ).clone();


  keypoints = makeKp3d(kps);
}

#if 0
void getOrbKp(
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors )
{
  //cv::Mat gray_img_small;
  //cv::resize( gray_img, gray_img_small, cv::Size(), 0.5, 0.5, CV_INTER_LINEAR);
  //cv::imshow("gray_img_small",gray_img_small);
  //cv::waitKey(100);

  cv::ORB::CommonParams p;
  p.first_level_ = 0;
  p.n_levels_ = 15;
  p.scale_factor_ = 1.4;
  cv::ORB orb( t, p );

//  explicit ORB(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31,
//                   int firstLevel = 0, int WTA_K=2, int scoreType=0, int patchSize=31 )

  /*
  int nfeatures = (int)(t);
  float scaleFactor = 1.2f;
  int nlevels = 30;
  int edgeThreshold = 31;
  int firstLevel = 0;
  int WTA_K=2;
  int scoreType=0;
  int patchSize=31;

  cv::ORB orb( nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize );
  */

  std::vector<cv::KeyPoint> kps;
  cv::Mat1b desc;

  cv::Mat empty_mask;

  //cv::imshow();

  orb( gray_img, empty_mask, kps, desc, false );

  std::cout << desc.type() << std::endl;
  std::cout << desc.rows << std::endl;
  std::cout << desc.cols << std::endl;
  std::cout << kps.size() << std::endl;

  descriptors.create( desc.rows, desc.cols*8 );
  for ( int i=0; i<desc.rows; i++ )
  {
    for ( int j=0; j<desc.cols; j++ )
    {
      //std::cout << int(desc(i,j)) << " = ";
      for ( unsigned k=0; k<8; k++ )
      {
        //std::cout << ((desc(i,j) >> k) & 1 );
        descriptors(i,j*8+k) = ((desc(i,j) >> k) & 1 );
      }
      //std::cout << std::endl;
    }
  }

  for ( unsigned i=0; i<kps.size(); i++ )
  {
    //std::cout << kps[i].octave << std::endl;
    kps[i].size = 6.0 * kps[i].octave;
    //kps[i].size *= 12.0/(float)patchSize;
    //kps[i].pt *= ;
    //std::cout << kps[i].size << std::endl;
  }

  keypoints = makeKp3d(kps);
}
#endif

void getDaftKp(
    daft_ns::DAFT::DetectorParams p_det,
    daft_ns::DAFT::DescriptorParams p_desc,
    const cv::Mat& gray_img,
    const cv::Mat1b& mask_img,
    const cv::Mat& depth_img,
    cv::Matx33f K,
    float t,
    std::vector<cv::KeyPoint3D>& keypoints,
    cv::Mat1f& descriptors )
{
  p_det.det_threshold_ *= t;

  daft_ns::DAFT daft1( p_det, p_desc );
  daft1( gray_img, mask_img, depth_img, K, keypoints, descriptors );
  /*

   p_det.base_scale_ *= sqrt(2);

   std::vector<cv::KeyPoint3D> kp2;
   daft_ns::DAFT daft2( p_det, p_desc );
   daft2.detect( gray_img, depth_img, K, kp2 );

   std::copy( kp1.begin(), kp1.end(), std::back_inserter(kp2) );

   return kp2;
   */
}
