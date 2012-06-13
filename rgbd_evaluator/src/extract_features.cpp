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

#include <stdio.h>

#include <sensor_msgs/image_encodings.h>

namespace rgbd_evaluator {

void sysCmd( std::string cmd )
{
  std::cout << "Executing system command: " << cmd << std::endl;

  if (system(cmd.c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << cmd << std::endl;
    std::cout << "--> check user permissions" << std::endl;
    exit(-1);
  }
}

ExtractDetectorFile::ExtractDetectorFile(std::string file_path,
    bool reset_files,
    bool verbose,
    bool small,
    int num_kp) {
  verbose_ = verbose;
  target_num_kp_ = num_kp;
  small_ = small;

  std::cout << "Starting extract_detector_file..." << std::endl;

  splitFileName(file_path);

  extra_folder_ = file_created_folder_ + "/kp_images";
  kp_folder_ = file_created_folder_ + "/keypoints";

  sysCmd("mkdir "+extra_folder_);
  sysCmd("mkdir "+kp_folder_);

  if ( reset_files )
  {
    sysCmd("rm "+extra_folder_+"/*.*");
    sysCmd("rm "+kp_folder_+"/*.*");
  }

  readDataFiles();
  extractAllKeypoints();
}

ExtractDetectorFile::~ExtractDetectorFile() {
  std::cout << "Stopping extract_detector_file..." << std::endl;
  cv::destroyAllWindows();
}

void ExtractDetectorFile::readDataFiles() {
  uint32_t i = 0;

  std::string path;
  path.append(file_created_folder_);
  path.append("/");
  path.append("K_");

  // read intrinsic matrix K once
  if (!readMatrix(path, K_)) {
    std::cout << path << " not found - aborting..." << std::endl;
    return;
  }

  std::string maskImagePath;
  maskImagePath.append(file_created_folder_);
  maskImagePath.append("/");
  maskImagePath.append("mask.pgm");

  // read mask image
  if (!fileExists(maskImagePath)) {
    std::cout << maskImagePath << " not found - aborting..." << std::endl;
    return;
  }

  cv::Mat mask_tmp = cv::imread(maskImagePath);

  mask_img_.create( mask_tmp.rows, mask_tmp.cols );

  for ( int y=0; y<mask_tmp.rows; y++ )
  {
    for ( int x=0; x<mask_tmp.cols; x++ )
    {
      if ( mask_tmp.at<cv::Vec3b>(y,x)[0] > 128 )
      {
        mask_img_(y,x) = 255;
      }
      else
      {
        mask_img_(y,x) = 0;
      }
    }
  }

  if ( small_ )
  {
    cv::resize( mask_img_.clone(), mask_img_, cv::Size(), 0.5, 0.5, CV_INTER_LINEAR );

    K_ = K_ * 0.5;
    K_(2,2) = 1;
  }


  //cv::imshow("mask_img_",mask_img_);

  // create image paths
  std::string image_rgb_name;
  image_rgb_name.append(file_created_folder_);
  image_rgb_name.append("/");
  image_rgb_name.append("img");

  std::string image_depth_name;
  image_depth_name.append(file_created_folder_);
  image_depth_name.append("/");
  image_depth_name.append("depth");

  // read rgb and depth images
  for (i = 0; i < 1000; i++) {
    std::cout << "Processing image: " << i + 1 << std::endl;

    std::stringstream ss;
    ss << i + 1;

    std::string tmp_rgb_name;
    tmp_rgb_name.append(image_rgb_name);
    tmp_rgb_name.append(ss.str());
    tmp_rgb_name.append(".ppm");

    std::string tmp_depth_name;
    tmp_depth_name.append(image_depth_name);
    tmp_depth_name.append(ss.str());
    tmp_depth_name.append(".pgm");

    if (!fileExists(tmp_rgb_name)) {
      std::cout << tmp_rgb_name << " not found! - aborting..." << std::endl;
      return;
    }

    if (!fileExists(tmp_depth_name)) {
      std::cout << tmp_depth_name << " not found! - aborting..." << std::endl;
      return;
    }

    ImageData img_data;

    // read homography
    if (i == 0) {
      img_data.hom = cv::Matx33f::eye();
    } else if (!readMatrix(file_created_folder_ + "/H1to" + ss.str() + "p",
        img_data.hom)) {
      std::cout << path << " not found - aborting..." << std::endl;
      return;
    }

    // read rgb and depth image and store data
    cv::Mat image_rgb = cv::imread(tmp_rgb_name);
    img_data.rgb_image = cv::Mat(image_rgb);

    // with read depth image
    cv::Mat1f depth_image;
    readDepth(tmp_depth_name, depth_image);

    img_data.depth_image = depth_image;

    if ( small_ )
    {
      cv::Matx33f scale_mat1 = cv::Matx33f::eye();
      cv::Matx33f scale_mat2 = cv::Matx33f::eye();
      scale_mat1(0,0) = scale_mat1(1,1) = 2.0;
      scale_mat2(0,0) = scale_mat2(1,1) = 0.5;
      img_data.hom = scale_mat2 * img_data.hom * scale_mat1;

      cv::resize( img_data.rgb_image.clone(), img_data.rgb_image, cv::Size(), 0.5, 0.5, CV_INTER_LINEAR );
      cv::resize( img_data.depth_image.clone(), img_data.depth_image, cv::Size(), 0.5, 0.5, CV_INTER_LINEAR );
    }

    image_store_.push_back(img_data);
  }
}

bool ExtractDetectorFile::fileExists(const std::string & fileName) {
  std::ifstream fileTest(fileName.c_str());

  if (!fileTest)
    return false;

  fileTest.close();
  return true;
}

bool ExtractDetectorFile::readMatrix(const std::string & fileName,
    cv::Matx33f& K) {
  static const uint32_t MATRIX_DIM = 3;

  // check if file exists
  if (!fileExists(fileName)) {
    std::cout << "ERROR: " << fileName << " not found!" << std::endl;
    return false;
  }

  // start reading data
  std::ifstream infile(fileName.c_str());

  K = cv::Matx33f(MATRIX_DIM, MATRIX_DIM);

  for (uint32_t y = 0; y < MATRIX_DIM; y++) {
    for (uint32_t x = 0; x < MATRIX_DIM; x++) {
      if (infile.eof()) {
        std::cout << "ERROR: end-of-file reached too early!" << std::endl;
        exit(-1);
      }
      float n;
      infile >> n;
      // write values to matrix
      K(y, x) = n;
    }
  }

  infile.close();
  return true;
}

bool ExtractDetectorFile::readDepth(const std::string & fileName,
    cv::Mat1f& depth_img) {
  uint32_t depth_rows = 0, depth_cols = 0;

  std::string input_string;
  std::ifstream infile;

  infile.open(fileName.c_str());
  getline(infile, input_string); // Header1

  if (input_string != "P2") {
    std::cout << fileName << ": Wrong image Header ( " << input_string
        << " ) ..." << std::endl;
    return false;
  }

  infile >> depth_cols;
  infile >> depth_rows;

  int maxval;
  infile >> maxval;

  depth_img = cv::Mat1f(depth_rows, depth_cols);

  for( uint32_t y = 0; y < depth_rows; y++ )
  {
    for ( uint32_t x = 0; x < depth_cols; x++ )
    {
      int n;
      if ( infile.eof() )
      {
        std::cout << "ERROR: end-of-file reached too early!" << std::endl;
        exit(-1);
      }
      infile >> n;
      if (n == 0) {
        depth_img(y, x) = std::numeric_limits<float>::quiet_NaN();
      } else {
        depth_img(y, x) = float(n) * 0.001;
      }
    }
  }

  infile.close();

  return true;
}

void ExtractDetectorFile::extractKeypoints(GetKpFunc getKp, std::string name, float t)
{
  uint32_t count = 1;
  bool first_image = true;

  cv::Mat first_kp_img;

  std::vector<ImageData>::iterator it;

  int it_step;
  std::vector<ImageData>::iterator it_end, it_begin;

  it_step = 1;
  it_begin = image_store_.begin();
  it_end = image_store_.end();

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = it_begin; it != it_end; it += it_step) {
    cv::Mat rgb_img = it->rgb_image;
    cv::Mat1f depth_img = it->depth_image;

#if 0
    double minval,maxval;
    cv::minMaxIdx( depth_img, &minval, &maxval );

    std::cout << "maxval "<<maxval << std::endl;

    cv::Mat1f tmp = depth_img.clone();
    tmp -= minval;
    tmp *= 1.0/(maxval-minval);
    cv::imshow( "Depth norm", tmp );
    cv::waitKey(100);
#endif

    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);

    std::cout << name << std::endl;

    if (it == it_begin && target_num_kp_ != 0)
    {
      // find optimal thresholds by secant method
      float t_left = t*0.5;
      float t_right = t*2;

      std::vector<cv::KeyPoint3D> kp_left,kp_right;
      cv::Mat1f descriptors;

      getKp(gray_img, mask_img_, depth_img, K_, t_left, kp_left, descriptors );
      getKp(gray_img, mask_img_, depth_img, K_, t_right, kp_right, descriptors );

      int y_left = kp_left.size() - target_num_kp_;
      int y_right = kp_right.size() - target_num_kp_;

      int its = 0;

      while (true)
      {
        float t_new;
        if ( y_left > 0 && y_right == -target_num_kp_ )
        {
          // if we've reached zero keypoints, take
          // middle
          t_new = (t_right + t_left) / 2;
        }
        else
        {
          // compute zero crossing of secant
          t_new = t_right - (float(t_right - t_left) / float(y_right - y_left) * float(y_right));
        }

        if ( t_new < 0 )
        {
          t_new = 0;
        }

        std::vector<cv::KeyPoint3D> kp_new;
        getKp(gray_img, mask_img_, depth_img, K_, t_new, kp_new, descriptors );
        int y_new = kp_new.size() - target_num_kp_;

        if ( std::abs(y_new) < 5 )
        {
          t = t_new;
          break;
        }

        std::cout << " y_left " << y_left << " y_right " << y_right << " y_new " << y_new << std::endl;
        std::cout << " t_left " << t_left << " t_right " << t_right << " t_new " << t_new << std::endl;

        if ( y_new > y_left )
        {
          y_right = y_left;
          y_left = y_new;
          t_right = t_left;
          t_left = t_new;
        }
        else if ( y_new >= y_right )
        {
          // we know that y_left < 0 and y_right > 0 !
          if ( y_new < 0 )
          {
            y_right = y_new;
            t_right = t_new;
          }
          else
          {
            y_left = y_new;
            t_left = t_new;
          }
        }
        else
        {
          y_left = y_right;
          y_right = y_new;
          t_left = t_right;
          t_right = t_new;
        }

        if ( !finite(t_right) ) {
          std::cout << "ERROR: cannot find enough keypoints!" << std::endl;
          getchar();
          exit(-1);
        }

        /*
         cv::Mat kp_img;
         cv::drawKeypoints3D(rgb_img, kp, kp_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
         cv::imshow("KP", kp_img);
         cv::waitKey(200);
         */
      }

      std::cout << std::endl;
      std::cout << "----------------------------------------";
      std::cout << std::endl;
      std::cout << name << ": t = " << t << std::endl;
      std::cout << std::endl;
      std::cout << "----------------------------------------";
      std::cout << std::endl;
    }

    std::stringstream s;
    s << "img" << count;
    count++;

    std::vector<cv::KeyPoint3D> kp;
    cv::Mat1f desc;

    static cv::Mat first_resp;
    static cv::Mat first_smooth;

    if (first_image) {
      getKp(gray_img, mask_img_, depth_img, K_, t, kp, desc );
      std::cout << "......................." << daft1.response_maps.size() << std::endl;

      std::cout << name << " " << s.str() << " #filtered kp = " << kp.size() << std::endl;

      first_image = false;

      cv::drawKeypoints3D(rgb_img, kp, first_kp_img, cv::Scalar(0, 0, 255),
          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      storeKeypoints(kp, desc, s.str(), name, rgb_img, rgb_img);

      if ( daft1.response_maps.size() > 0 )
      {
        first_resp = daft1.response_maps.begin()->second;
        first_smooth = daft1.smoothed_imgs.begin()->second;
        //cv::drawKeypoints3D(daft1.response_maps.begin()->second * 2.0 + 0.5, kp, resp, cv::Scalar(0, 0, 255),
        //    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow( "daft1.response_maps[0] warped", first_resp * 2.0 + 0.5 );
        cv::imshow( "daft1.smoothed_imgs[0] warped", first_smooth );
      }
    } else {

      /*
      // warp first image into current (hack)
      cv::warpPerspective( image_store_.begin()->rgb_image, rgb_img, cv::Mat(it->hom),
          cv::Size(rgb_img.cols, rgb_img.rows), 0, cv::INTER_LINEAR, cv::BORDER_TRANSPARENT );
      cv::cvtColor(rgb_img, gray_img, CV_BGR2GRAY);
      */

      cv::Mat1b fake_mask( (int)gray_img.rows, (int)gray_img.cols, (cv::Mat1b::value_type)1 );
      getKp(gray_img, fake_mask, depth_img, K_, t, kp, desc );
      std::cout << name << " " << s.str() << " #kp = " << kp.size() << std::endl;

      cv::Mat first_kp_img_warped = rgb_img.clone();
      cv::warpPerspective(first_kp_img, first_kp_img_warped, cv::Mat(it->hom),
          cv::Size(rgb_img.cols, rgb_img.rows));

      if ( daft1.response_maps.size() > 0 )
      {
        cv::Mat resp_warped,smooth_warped;
        cv::warpPerspective(daft1.response_maps.begin()->second, resp_warped, cv::Mat(it->hom.inv()),
            cv::Size(rgb_img.cols, rgb_img.rows));
        cv::warpPerspective(daft1.smoothed_imgs.begin()->second, smooth_warped, cv::Mat(it->hom.inv()),
            cv::Size(rgb_img.cols, rgb_img.rows));
        cv::imshow( "daft1.response_maps[0] warped", resp_warped*2+0.5 );
        cv::imshow( "daft1.smoothed_imgs[0] warped", smooth_warped );
        cv::imshow( "daft1.response_maps[0] diff", (resp_warped-first_resp)*2+0.5 );
        cv::imshow( "daft1.smoothed_imgs[0] diff", smooth_warped-first_smooth );
      }

      storeKeypoints(kp, desc, s.str(), name, rgb_img, first_kp_img_warped);
    }

    if (verbose_) {
      std::cout << "Press any key to continue." << std::endl;
      while (cv::waitKey(100) == -1)
        ;
    }
  }
}

void ExtractDetectorFile::extractAllKeypoints()
{
  cv::daft2::DAFT::DetectorParams det_p;
  cv::daft2::DAFT::DescriptorParams desc_p;
  //p.max_px_scale_ = 800;
  //det_p.min_px_scale_ = 3.0;
  //det_p.base_scale_ = 0.025;
  //det_p.scale_levels_ = 1;
  //det_p.max_princ_curv_ratio_ = 0.0;
  //det_p.det_threshold_ = 0.0;
  //desc_p.z_thickness_ = 0.3;

  det_p.affine_multiscale_ = false;

  det_p.det_type_=det_p.DET_GAUSS3D;
  det_p.affine_=true;
  det_p.max_search_algo_ = det_p.MAX_WINDOW;
  //desc_p.octave_offset_ = -1;
  extractKeypoints( boost::bind( &getDaftKp, det_p, desc_p, _1,_2,_3,_4,_5,_6,_7 ), "DAFT Gauss3D", 1.0 ); //1.44445);

  det_p.det_type_=det_p.DET_FELINE;
  det_p.affine_=true;
  det_p.max_search_algo_ = det_p.MAX_WINDOW;
  //desc_p.octave_offset_ = -1;
  extractKeypoints( boost::bind( &getDaftKp, det_p, desc_p, _1,_2,_3,_4,_5,_6,_7 ), "DAFT", 1.0 ); //1.44445);

  extractKeypoints( &getSurfKp, "SURF", 21.8753 );
  extractKeypoints( &getSiftKp, "SIFT", 3.09896 );
  //extractKeypoints( &getOrbKp, "ORB", target_num_kp_ );

  det_p.affine_=false;
  //extractKeypoints( boost::bind( &getDaftKp, det_p, desc_p, _1,_2,_3,_4, _5, _6, _7 ), "DAFT Non-Affine", 3.14 );

}

void ExtractDetectorFile::printMat(cv::Matx33f M) {
  std::cout << std::setprecision(3) << std::right << std::fixed;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      std::cout << std::setw(5) << (double) M(row, col) << " ";
    }
    std::cout << std::endl;
  }
}

std::vector<cv::KeyPoint3D> ExtractDetectorFile::filterKpMask(
    std::vector<cv::KeyPoint3D> kp) {
  // filter keypoints which dont fit the mask
  std::vector<cv::KeyPoint3D> kp_filtered;

  for (uint32_t i = 0; i < kp.size(); i++) {
    //std::cout << int(mask_img_( kp.at(i).pt.y, kp.at(i).pt.x )) << " ";
    // check for black spots in maskimage
    if (mask_img_.at<cv::Vec3b>(int(kp[i].pt.y), int(kp[i].pt.x))[0] > 128) {
      kp_filtered.push_back(kp[i]);
    }
  }

  std::cout << kp_filtered.size() << "   of " << kp.size() << " keypoints within mask boundaries." << std::endl;
  std::cout << std::endl;
  return kp_filtered;
}

void ExtractDetectorFile::storeKeypoints(
    std::vector<cv::KeyPoint3D> keypoints,
    cv::Mat1f& descriptors,
    std::string img_name,
    std::string extension,
    cv::Mat& rgb_img,
    cv::Mat& warped_img )
{
  if (keypoints.size() == 0) {
    return;
  }

  // draw keypoints
  cv::Mat kp_img,warped_kp_img;

  cv::drawKeypoints3D(warped_img, keypoints, warped_kp_img, cv::Scalar(255, 255, 255),
      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints3D(rgb_img, keypoints, kp_img, cv::Scalar(255, 255, 255),
      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  /*
  cv::putText(kp_img, extension, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1,
      cv::Scalar(0, 0, 0), 5, CV_AA);
  cv::putText(kp_img, extension, cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1,
      cv::Scalar(255, 255, 255), 2, CV_AA);
  */

  if (verbose_) {
    cv::imshow("kp", kp_img);
    cv::imshow("kp warped", warped_kp_img);
    cv::waitKey(50);
    return;
  }

  /*
   std::stringstream s;
   s.width(3);
   s.fill('0');
   s << img_count;
   std::string img_file_name = extra_folder_ + "/" + s.str() + ".ppm";
   img_count++;
   */

  std::string img_file_name = extra_folder_ + "/" + extension + "_" + img_name
      + ".ppm";
  std::cout << "Writing " << img_file_name << std::endl;
  cv::imwrite(img_file_name, kp_img);

  std::string warped_img_file_name = extra_folder_ + "/warped " + extension + "_" + img_name
      + ".ppm";
  std::cout << "Writing " << warped_img_file_name << std::endl;
  cv::imwrite(warped_img_file_name, warped_kp_img);


  // store keypoint files

  std::vector<cv::KeyPoint3D>::iterator it;
  int k;

  double_t ax, bx, ay, by, a_length, b_length, alpha_a, alpha_b;
  double_t A, B, C;

  std::string filePath = kp_folder_ + "/" + img_name + "."
      + extension;

  // open file
  std::cout << "Writing keypoints to " << filePath.c_str() << std::endl;
  std::fstream file;
  file.open(filePath.c_str(), std::ios::out);

  // header
  file << descriptors.cols + 1 << std::endl;
  file << keypoints.size() << std::endl;

  for (k=0,it = keypoints.begin(); it != keypoints.end(); k++,it++) {
    //hack
    //it->affine_minor = it->affine_major;

    ax = cos(it->aff_angle);
    ay = sin(it->aff_angle);
    bx = -ay;
    by = ax;

    alpha_a = atan2(ay, ax);
    alpha_b = atan2(by, bx);

    a_length = 0.5 * it->aff_major;
    b_length = 0.5 * it->aff_minor;

    ax = cos(alpha_a);
    bx = cos(alpha_b);
    ay = sin(alpha_a);
    by = sin(alpha_b);

    A = (pow(ax, 2) * pow(b_length, 2) + pow(bx, 2) * pow(a_length, 2))
        / (pow(a_length, 2) * pow(b_length, 2));

    B = ((ax * ay * pow(b_length, 2) + bx * by * pow(a_length, 2)))
        / (pow(a_length, 2) * pow(b_length, 2));

    C = (pow(ay, 2) * pow(b_length, 2) + pow(by, 2) * pow(a_length, 2))
        / (pow(a_length, 2) * pow(b_length, 2));

    file << it->pt.x << "  " << it->pt.y << "  " << A << "  " << B << "  " << C;

    // write world scale as part of the feature vector,
    // so keypoints of different size don't get matched
    if (it->world_size != 0) {
      float s_log = log2(it->world_size);
      file << " " << s_log * 100;
    } else {
      file << " 0.0";
    }

    for (unsigned i = 0; i < descriptors.cols; i++) {
      file << " " << descriptors(k,i);
    }

    file << std::endl;

  }

  file.close();
}

void ExtractDetectorFile::splitFileName(const std::string& str) {
  size_t found;
  std::cout << "Splitting: " << str << std::endl;
  found = str.find_last_of("/\\");

  file_path_ = str.substr(0, found);
  file_name_ = str.substr(found + 1);

  found = file_name_.find_last_of(".");
  file_folder_ = file_name_.substr(0, found);

  file_created_folder_.append(file_path_);
  file_created_folder_.append("/");
  file_created_folder_.append(file_folder_);

  std::cout << " path: " << file_path_ << std::endl;
  std::cout << " file: " << file_name_ << std::endl;
  std::cout << " folder: " << file_folder_ << std::endl;
  std::cout << " created folder: " << file_created_folder_ << std::endl;
}

} /* namespace rgbd_evaluator */

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Wrong usage, Enter: " << argv[0]
        << "[-v] [-k num_kp] <bagfile1> <bagfile2> .." << std::endl;
    return -1;
  }

  bool verbose = false;
  bool reset_files = false;
  bool small = false;
  int num_kp = 0;
  std::vector<std::string> bagfiles;

  for (int i = 1; i < argc; i++) {
    std::string arg = std::string(argv[i]);
    if (arg == "-v") {
      std::cout << "Verbose on" << std::endl;
      verbose = true;
    } else if (arg == "-r") {
      std::cout << "Deleting old output files!" << std::endl;
      reset_files = true;
    } else if (arg == "-s") {
      std::cout << "Using half-size images!" << std::endl;
      small = true;
    } else if (i < argc - 1 && arg == "-k") {
      num_kp = atoi(argv[i + 1]);
      std::cout << "num_kp = " << num_kp << std::endl;
      i++;
    } else {
      bagfiles.push_back(arg);
      std::cout << "using bagfile: " << arg << std::endl;
    }
  }

  std::cout << "num_kp = " << num_kp << std::endl;
  std::cout << "----------------------------------------" << std::endl << std::endl;

  for (unsigned i = 0; i < bagfiles.size(); i++) {
    rgbd_evaluator::ExtractDetectorFile extract_detector_file(bagfiles[i],
        reset_files, verbose, small, num_kp);
  }

  std::cout << "Exiting.." << std::endl;
  return 0;
}

