/*
 * extract_detector_file.cpp
 *
 *  Created on: Feb 20, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/extract_detector_file.h"

#include <iostream>
#include <fstream>

#include <math.h>
#include <boost/timer.hpp>

#include <sensor_msgs/image_encodings.h>

namespace rgbd_evaluator
{

const int num_kp = 750;

ExtractDetectorFile::ExtractDetectorFile(std::string file_path, bool reverse_order)
{
  std::cout << "Starting extract_detector_file..." << std::endl;

  reverse_order_ = reverse_order;

  splitFileName(file_path);

  // create folder to store detector file
  std::string makeFolder;
  makeFolder.append("mkdir ");
  makeFolder.append(file_created_folder_);

  if( system(makeFolder.c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << makeFolder  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  extra_folder_ = file_created_folder_ + "/extra";

  if( system(("mkdir "+ extra_folder_).c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << "mkdir "+ extra_folder_  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  bag_.open(file_path, rosbag::bagmode::Read);

  readBagFile();

  extractAllKeypoints();
}

ExtractDetectorFile::~ExtractDetectorFile()
{
  std::cout << "Stopping extract_detector_file..." << std::endl;

  bag_.close();
}

void ExtractDetectorFile::readBagFile()
{
  bool got_cam_info = false;

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back("rgb_img");
  topics.push_back("depth_img");
  topics.push_back("cam_info");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  image_store_.push_back( ImageData() );

  // Load all messages into our stereo dataset
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // if the current image data is complete, go to next one
      if ( image_store_.back().isComplete() )
      {
        image_store_.push_back( ImageData() );
      }

      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();

      //check if rgb_img message arrived
      if (p_rgb_img != NULL && p_rgb_img->encoding == "bgr8" )
      {
        if ( image_store_.back().rgb_image )
        {
          std::cout << "There is already an rgb image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);

        // store data in vectorImageData
        image_store_.back().rgb_image = ptr;
      }

      /**********************************************************************************************************************/

      // load depth image
      sensor_msgs::Image::ConstPtr p_depth_img = m.instantiate<sensor_msgs::Image>();

      //check if depth_img message arrived
      if (p_depth_img != NULL && p_depth_img->encoding == "32FC1" )
      {
        if ( image_store_.back().depth_image )
        {
          std::cout << "There is already an depth image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_depth_img);

        // store data in vectorImageData
        image_store_.back().depth_image = ptr;
      }

      /**********************************************************************************************************************/

      // load cam_info
      sensor_msgs::CameraInfo::ConstPtr p_cam_info = m.instantiate<sensor_msgs::CameraInfo>();

      if(( p_cam_info != NULL ) && ( got_cam_info == false ))
      {
        boost::array<double,9> cam_info = p_cam_info->K;

        K_ = cv::Matx33f(cam_info.at(0), cam_info.at(1), cam_info.at(2),
                         cam_info.at(3), cam_info.at(4), cam_info.at(5),
                         cam_info.at(6), cam_info.at(7), cam_info.at(8));

        got_cam_info = true;
      }
  }

  // if the current image data is complete, go to next one
  if ( !image_store_.back().isComplete() )
  {
    image_store_.erase( image_store_.end()-1 );
  }

}


std::vector<cv::KeyPoint3D> makeKp3d( std::vector<cv::KeyPoint> kp )
{
  std::vector<cv::KeyPoint3D> kp_3d;
  kp_3d.reserve( kp.size() );
  for ( size_t i=0; i < kp.size(); i++ )
  {
    kp_3d.push_back( kp[i] );
  }
  return kp_3d;
}

// these helper function compute the number of keypoints
// for a given threshold
std::vector<cv::KeyPoint3D> getSurfKp( const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  cv::SURF surf( t * 100.0 );
  cv::Mat mask;
  std::vector<cv::KeyPoint> kp;
  surf( gray_img, mask, kp );
  return makeKp3d( kp );
}
std::vector<cv::KeyPoint3D> getSiftKp( const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  cv::SIFT::CommonParams cp;
  cv::SIFT::DetectorParams detp;
  cv::SIFT::DescriptorParams descp;
  detp.threshold = detp.GET_DEFAULT_THRESHOLD() * t;
  //detp.edgeThreshold = detp.GET_DEFAULT_EDGE_THRESHOLD() * t;
  cv::SIFT sift = cv::SIFT( cp, detp, descp );
  std::vector<cv::KeyPoint> kp;
  cv::Mat mask;
  sift( gray_img, mask, kp );
  return makeKp3d( kp );
}

/*
std::vector<cv::KeyPoint3D> getDaftKp( cv::DAFT::DetectorParams p, const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  std::vector<cv::KeyPoint3D> kp;
  p.det_threshold_ *= t;
  cv::DAFT daft( p );
  daft.detect( gray_img, depth_img, K, kp );
  return kp;
}
*/

std::vector<cv::KeyPoint3D> getDaftKp( cv::DAFT::DetectorParams p, const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  std::vector<cv::KeyPoint3D> kp;
  p.det_threshold_ *= t;
  cv::DAFT daft( p );
  daft.detect( gray_img, depth_img, K, kp );
  return kp;
}


void ExtractDetectorFile::extractKeypoints( GetKpFunc getKp, std::string name )
{
  uint32_t count = 1;

  std::vector< ImageData >::iterator it;

  int it_step;
  std::vector< ImageData >::iterator it_end,it_begin;
  if ( reverse_order_ )
  {
    it_step = -1;
    it_begin = image_store_.end()-1;
    it_end = image_store_.begin()-1;
  }
  else
  {
    it_step = 1;
    it_begin = image_store_.begin();
    it_end = image_store_.end();
  }

  float t=1;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = it_begin; it != it_end; it+=it_step)
  //it = it_begin;
  {
    cv::Mat bag_rgb_img = it->rgb_image.get()->image;
    cv::Mat bag_depth_img = it->depth_image.get()->image;

    cv::Mat rgb_img = bag_rgb_img;
    cv::Mat depth_img = bag_depth_img;

    int scale_fac = bag_rgb_img.cols / bag_depth_img.cols;

#if 1
    // Resize depth to have the same width as rgb
    cv::resize( bag_depth_img, depth_img, cvSize(0,0), scale_fac, scale_fac, cv::INTER_LINEAR );

    // Crop rgb so it has the same size as depth
    rgb_img = cv::Mat( bag_rgb_img, cv::Rect( 0,0, depth_img.cols, depth_img.rows ) );
#else
    depth_img = bag_depth_img;
    // make intensity image smaller to have the same aspect ratio and size as depth
    cv::Mat tmp1 = cv::Mat( bag_rgb_img, cv::Rect( 0,0, depth_img.cols*scale_fac, depth_img.rows*scale_fac ) );
    cv::resize( tmp1, rgb_img, cvSize(depth_img.cols, depth_img.rows) );

    static bool b=0;
    if ( !b )
    {
      K_(0,0)/=2.0;
      K_(1,1)/=2.0;
      K_(0,2)/=2.0;
      K_(1,2)/=2.0;
      b=1;
    }
#endif

#if 0
    double minval,maxval;
    cv::minMaxIdx( depth_img, &minval, &maxval );

    cv::Mat tmp = depth_img.clone();
    tmp -= minval;
    tmp *= 1.0/(maxval-minval);
    cv::imshow( "Depth", tmp );
    cv::waitKey(100);
#endif

    cv::Mat gray_img;
    cv::cvtColor( rgb_img, gray_img, CV_BGR2GRAY );

    cv::Mat mask;

    std::cout << name << std::endl;

    if ( it == it_begin )
    {
      // find optimal thresholds by secant method
      float last_t=1;
      int last_kp_size = 0;
      int its=0;
      int kp_size = 0;

      while ( ( kp_size == 0 ) || ( std::abs(kp_size - num_kp) > 10 ) )
      {
        last_kp_size = kp_size;
        kp_size = getKp( gray_img, depth_img, K_, t ).size();

        std::cout << " t_" << its-1 << " = " << last_t << " f(t_n-1) " << last_kp_size << std::endl;
        std::cout << " t_" << its << " = " << t << " f(t_n)=" << kp_size << std::endl;

        // first iteration: guess step width
        if ( its == 0 )
        {
          float ratio = float(kp_size) / float(num_kp);
          last_t = t;
          t *= 1.0 + 0.5 * (ratio-1.0);
        }
        else
        {
          // compute zero crossing of secant
          float t_next = t - ( float(t-last_t) / float(kp_size-last_kp_size) * float(kp_size-num_kp) );
          last_t = t;
          t = t_next;
        }
        std::cout << " t_" << its+1 << " = " << t << std::endl;

        its++;
        std::cout << std::endl;
      }
    }

    std::stringstream s;
    s << "img" << count;
    count++;

#if 0
    for ( int i=0; i<3; i++ )
    {
      daft.detect(gray_img, depth_img, K_, kp);
    }
    {
      boost::timer timer;
      timer.restart();
      for ( int i=0; i<10; i++ )
      {
        daft.detect(gray_img, depth_img, K_, kp);
      }
      std::cout << name << " execution time [ms]: " << timer.elapsed()*100 << std::endl;
    }
#else
    std::vector<cv::KeyPoint3D> kp = getKp( gray_img, depth_img, K_, t );
    std::cout << name << " " << s.str() << " #kp = " << kp.size() << std::endl;

    storeKeypoints(kp, s.str(), name, rgb_img );
#endif

#if 0
    std::cout << "Press any Key to continue!" << std::endl;
    getchar();
#endif
  }
}

void ExtractDetectorFile::extractAllKeypoints()
{
  cv::DAFT::DetectorParams p;
  p.max_px_scale_ = 500;
  p.min_px_scale_ = 2;
  //p.base_scale_ = 0.02;
  //p.scale_levels_ = 1;
  p.det_threshold_ = 0.1;//115;
  p.pf_threshold_ = 10;

  p.det_type_=p.DET_DOB;
  p.affine_=false;
  p.max_search_algo_ = p.MAX_FAST;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT-Fast" );

  p.det_type_=p.DET_DOB;
  p.affine_=true;
  p.max_search_algo_ = p.MAX_WINDOW_AFFINE;
  extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT-Fast Affine" );

  p.det_type_ = p.DET_LAPLACE;
  p.max_search_algo_ = p.MAX_FAST;
  p.affine_ = false;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT" );

  p.det_type_ = p.DET_LAPLACE;
  p.max_search_algo_ = p.MAX_WINDOW_AFFINE;
  p.affine_ = true;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT Affine" );

  //extractKeypoints( &getSurfKp, "SURF" );
  //extractKeypoints( &getSiftKp, "SIFT" );
}

void ExtractDetectorFile::storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, std::string img_name, std::string extension, cv::Mat& rgb_img )
{
  std::vector< cv::KeyPoint3D >::iterator it;
  double_t ax, bx, ay, by, a_length, b_length, alpha_a, alpha_b;
  double_t A, B, C;

  std::string filePath = file_created_folder_ +  "/" + img_name + "." +extension;

  // open file
  std::fstream file;
  file.open(filePath.c_str(), std::ios::out);

  // header
  file << "1.0" << std::endl;
  file << keypoints.size() << std::endl;

  for ( it = keypoints.begin(); it != keypoints.end(); it++ )
  {
    ax = cos( it->affine_angle );
    ay = sin( it->affine_angle );
    bx = -ay;
    by = ax;

    alpha_a = atan2(ay,ax);
    alpha_b = atan2(by,bx);

    a_length = it->affine_major;
    b_length = it->affine_minor;

    ax = cos(alpha_a);
    bx = cos(alpha_b);
    ay = sin(alpha_a);
    by = sin(alpha_b);

    A = ( pow(ax,2) * pow(b_length,2) + pow(bx,2) * pow(a_length,2)) / (pow(a_length,2) * pow(b_length,2) );

    B = 2 * ( ( ax * ay * pow(b_length,2) + bx * by * pow(a_length,2)) ) / (pow(a_length,2) * pow(b_length,2) );

    C = ( pow(ay,2) * pow(b_length,2) + pow(by,2) * pow(a_length,2)) / (pow(a_length,2) * pow(b_length,2) );

    file << it->pt.x << "  " << it->pt.y << "  " << A << "  " << B << "  " << C << std::endl;

  }

  file.close();

  // draw keypoints
  cv::Mat kp_img;

  cv::drawKeypoints3D(rgb_img, keypoints, kp_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  cv::putText( kp_img, extension, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 5, CV_AA );
  cv::putText( kp_img, extension, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 2, CV_AA );

  cv::imshow(extension, kp_img);
  cv::waitKey(100);

  cv::imwrite( extra_folder_ + "/" + extension + " " + img_name + ".ppm", kp_img);
}

void ExtractDetectorFile::splitFileName(const std::string& str)
{
  size_t found;
  std::cout << "Splitting: " << str << std::endl;
  found=str.find_last_of("/\\");

  file_path_ = str.substr(0,found);
  file_name_ = str.substr(found+1);

  found = file_name_.find_last_of(".");
  file_folder_ = file_name_.substr(0,found);

  file_created_folder_.append(file_path_);
  file_created_folder_.append("/");
  file_created_folder_.append(file_folder_);

  std::cout << " path: " << file_path_ << std::endl;
  std::cout << " file: " << file_name_ << std::endl;
  std::cout << " folder: " << file_folder_ << std::endl;
  std::cout << " created folder: " << file_created_folder_ << std::endl;
}

} /* namespace rgbd_evaluator */




int main( int argc, char** argv )
{
  if(argc < 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << " <bagfileName>" << std::endl;
    return -1;
  }

  std::string file_name(argv[1]);

  bool reverse_order = argc > 2 && std::string(argv[2]) == "-r";
  std::cout << "reverse_order " << reverse_order << std::endl;

  rgbd_evaluator::ExtractDetectorFile extract_detector_file(file_name, reverse_order);

  std::cout << "Exiting.." << std::endl;
  return 0;
}

