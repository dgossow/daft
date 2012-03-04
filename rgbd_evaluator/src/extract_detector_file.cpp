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

  bag_.open(file_path, rosbag::bagmode::Read);

  readBagFile();

  extractKeypoints();
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


void ExtractDetectorFile::extractDaftKeypoints( cv::DAFT::DetectorParams p, std::string name )
{
  uint32_t count = 1;

  std::vector< ImageData >::iterator it;

  cv::DAFT daft( p );

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

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = it_begin; it != it_end; it+=it_step)
  {
    std::vector<cv::KeyPoint3D> daft_kp;

    cv::Mat bag_rgb_img = it->rgb_image.get()->image;
    cv::Mat bag_depth_img = it->depth_image.get()->image;

    cv::Mat rgb_img;
    cv::Mat depth_img;

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
#endif

    cv::GaussianBlur( depth_img, depth_img, cv::Size(), 2, 2 );

    double minval,maxval;
    cv::minMaxIdx( depth_img, &minval, &maxval );

    cv::Mat tmp = depth_img.clone();
    tmp -= minval;
    tmp *= 1.0/(maxval-minval);
    cv::imshow( "Depth", tmp );
    cv::waitKey(100);

    cv::Mat gray_img;
    cv::cvtColor( rgb_img, gray_img, CV_BGR2GRAY );

    cv::Mat mask;

    const int num_kp = 750;

    float det_t_init = p.det_threshold_;
    float det_pf_init = p.pf_threshold_;

    if ( it == it_begin )
    {
      float t=1;
      while ( daft_kp.size() == 0 || std::abs(int(daft_kp.size()) - num_kp ) > 10 )
      {
        p.det_threshold_ = det_t_init * t;
        p.pf_threshold_ = det_pf_init * t;
        //p.pf_threshold_ = 0.001 * t;
        daft_kp.clear();
        daft = cv::DAFT( p );

        daft.detect(gray_img, depth_img, K_, daft_kp);

        std::cout << name << " kp " << daft_kp.size() << std::endl;

        float ratio = float(daft_kp.size()) / float(num_kp);

        t *= 1.0 + 2.5 * (ratio-1.0);
        std::cout << "ratio " << ratio << " t " << t << " p.pf_threshold_ " << p.pf_threshold_ << std::endl;
        std::cout << " p.det_threshold_ " << p.det_threshold_ << std::endl;
      }
    }

    daft.detect(gray_img, depth_img, K_, daft_kp);
    std::cout << "daft_kp " << daft_kp.size() << std::endl;

    // draw keypoints
    cv::Mat daft_img;

    cv::drawKeypoints3D(rgb_img, daft_kp, daft_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow(name, daft_img);

    cv::waitKey(100);

    std::stringstream s;
    s << "img" << count;
    count++;

    storeKeypoints(daft_kp, s.str(), name );

#if 0
    std::cout << "Press any Key to continue!" << std::endl;
    getchar();
#endif
  }

}



void ExtractDetectorFile::extractKeypoints()
{
  uint32_t count = 1;

  std::vector< ImageData >::iterator it;

  cv::DAFT::DetectorParams p;
  p.max_px_scale_ = 300;
  p.min_px_scale_ = 3;
  p.base_scale_ = 0.0125;
  p.scale_levels_ = 5;
  p.affine_=false;
  p.pf_threshold_ = 0.0001;
  extractDaftKeypoints( p, "DAFT" );


  cv::DAFT::DetectorParams p_affine=p;
  p_affine.affine_=true;
  extractDaftKeypoints( p_affine, "DAFT affine" );

  cv::DAFT::DetectorParams p_laplace=p;
  p_laplace.det_type_ = p.DET_LAPLACE;
  p_laplace.pf_type_ = p.PF_PRINC_CURV_RATIO;
  p_laplace.max_search_algo_ = p_laplace.MAX_WINDOW_AFFINE;
  p_laplace.pf_threshold_ = 10;

  extractDaftKeypoints( p_laplace, "DAFT Laplace" );

  cv::SIFT sift;
  cv::SURF surf;

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

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = it_begin; it != it_end; it+=it_step)
  {
    std::vector<cv::KeyPoint> sift_kp,surf_kp;

    cv::Mat bag_rgb_img = it->rgb_image.get()->image;
    cv::Mat bag_depth_img = it->depth_image.get()->image;

    cv::Mat rgb_img;
    cv::Mat depth_img;

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
#endif

    cv::GaussianBlur( depth_img, depth_img, cv::Size(), 2, 2 );

    double minval,maxval;
    cv::minMaxIdx( depth_img, &minval, &maxval );

    cv::Mat tmp = depth_img.clone();
    tmp -= minval;
    tmp *= 1.0/(maxval-minval);
    cv::imshow( "Depth", tmp );
    cv::waitKey(100);

    cv::Mat gray_img;
    cv::cvtColor( rgb_img, gray_img, CV_BGR2GRAY );

    cv::Mat mask;

    const int num_kp = 750;

    if ( it == it_begin )
    {
      double t=1;
      while ( sift_kp.size() == 0 || std::abs(sift_kp.size() - num_kp ) > 10 )
      {
        sift_kp.clear();
        cv::SIFT::CommonParams cp;
        cv::SIFT::DetectorParams detp;
        cv::SIFT::DescriptorParams descp;
        detp.threshold = detp.GET_DEFAULT_THRESHOLD() * t;
        sift = cv::SIFT( cp, detp, descp );

        std::cout << "edge_t " << detp.edgeThreshold << std::endl;

        sift( gray_img, mask, sift_kp );
        std::cout << "sift_kp " << sift_kp.size() << std::endl;

        float ratio = float(sift_kp.size()) / float(num_kp);

        t *= 1.0 + 0.5 * (ratio-1.0);
        std::cout << "ratio " << ratio << " t " << t << std::endl;
      }
      t=1;
      while ( surf_kp.size() == 0 || std::abs(surf_kp.size() - num_kp ) > 10 )
      {
        surf_kp.clear();
        surf = cv::SURF( t * 100.0 );

        surf( gray_img, mask, surf_kp );
        std::cout << "surf_kp " << surf_kp.size() << std::endl;

        float ratio = float(surf_kp.size()) / float(num_kp);

        t *= 1.0 + 0.5 * (ratio-1.0);
        std::cout << "ratio " << ratio << " t " << t << std::endl;
      }
      t=1;
#if 0
      // compare speeds
      for ( int i=0; i<3; i++ )
      {
        surf( gray_img, mask, surf_kp );
      }
      {
        boost::timer timer;
        timer.restart();
        for ( int i=0; i<10; i++ )
        {
          surf( gray_img, mask, surf_kp );
        }
        std::cout << "surf execution time [ms]: " << timer.elapsed()*100 << std::endl;
      }
      for ( int i=0; i<3; i++ )
      {
        sift( gray_img, mask, sift_kp );
      }
      {
        boost::timer timer;
        timer.restart();
        for ( int i=0; i<10; i++ )
        {
          sift( gray_img, mask, sift_kp );
        }
        std::cout << "sift execution time [ms]: " << timer.elapsed()*100 << std::endl;
      }
#endif
    }
    sift( gray_img, mask, sift_kp );
    std::cout << "sift_kp " << sift_kp.size() << std::endl;
    surf( gray_img, mask, surf_kp );
    std::cout << "surf_kp " << surf_kp.size() << std::endl;


    // draw keypoints
    cv::Mat sift_img,surf_img;

    cv::drawKeypoints(rgb_img, sift_kp, sift_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::drawKeypoints(rgb_img, surf_kp, surf_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("SIFT Keypoints", sift_img);
    cv::imshow("SURF Keypoints", surf_img);

    cv::waitKey(100);

    std::stringstream s;
    s << "img" << count;
    count++;

    storeKeypoints(sift_kp, s.str(), "SIFT" );
    storeKeypoints(surf_kp, s.str(), "SURF" );

#if 1
    std::cout << "Press any Key to continue!" << std::endl;
    getchar();
#endif
  }

}

void ExtractDetectorFile::storeKeypoints(std::vector<cv::KeyPoint> keypoints, std::string img_name, std::string extension )
{
  std::vector<cv::KeyPoint3D> keypoints_3d;
  keypoints_3d.reserve( keypoints.size() );
  for ( size_t i=0; i < keypoints.size(); i++ )
  {
    keypoints_3d.push_back( keypoints[i] );
  }
  storeKeypoints( keypoints_3d, img_name, extension );
}

void ExtractDetectorFile::storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, std::string img_name, std::string extension )
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

