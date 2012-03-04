/*
 * rgbd_evaluator_preprocessing.cpp
 *
 *  Created on: Jan 13, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/rgbd_evaluator_preprocessing.h"

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <LinearMath/btQuaternion.h>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btVector3.h>

namespace rgbd_evaluator
{

RgbdEvaluatorPreprocessing::RgbdEvaluatorPreprocessing(std::string file_path, bool reverse_order)
{
  std::cout << "Reading bagfile from " << file_path.c_str() << std::endl;

  reverse_order_ = reverse_order;

  splitFileName(file_path);

  // create folder to store images and homography --> should be in /home/... otherwise root permissions neccessary
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

}

RgbdEvaluatorPreprocessing::~RgbdEvaluatorPreprocessing()
{
  std::cout << "Stopping preprocessing..." << std::endl;

  cv::destroyAllWindows();

  bag_.close();
}

void RgbdEvaluatorPreprocessing::createTestFiles()
{
  uint32_t count = 1;
  bool got_cam_info = false;

  // Image topics to load
  std::vector<std::string> topics;
  topics.push_back("rgb_img");
  topics.push_back("depth_img");
  topics.push_back("center_transform");
  topics.push_back("cam_info");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  sensor_msgs::Image::ConstPtr p_current_img;
  geometry_msgs::TransformStamped::ConstPtr p_current_transform;
  sensor_msgs::CameraInfo::ConstPtr p_cam_info;

  image_store_.push_back( ImageData() );

  // Load all messages into our stereo dataset
  BOOST_FOREACH(rosbag::MessageInstance const m, view)
  {
      // if the current image data is complete, go to next one
      if ( image_store_.back().isComplete() )
      {
        image_store_.push_back( ImageData() );
        //std::cout << "ImageData_"<< count << " complete! Press any key to continue\n\r" << std::endl;
        //getchar();
        count++;
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

      // load center transform
      geometry_msgs::TransformStamped::ConstPtr p_center_transform = m.instantiate<geometry_msgs::TransformStamped>();

      if ( p_center_transform != NULL )
      {
        //std::cout << "center_transform available" << std::endl;

        if ( image_store_.back().approx_transform )
        {
          std::cout << "There is already a transform for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        tf::StampedTransform transform;
        tf::transformStampedMsgToTF(*p_center_transform, transform);

        image_store_.back().approx_transform = boost::make_shared<btTransform>(transform);
      }

      /**********************************************************************************************************************/

      // load cam_info
      sensor_msgs::CameraInfo::ConstPtr p_cam_info = m.instantiate<sensor_msgs::CameraInfo>();

      if(( p_cam_info != NULL ) && ( got_cam_info == false ))
      {
        //std::cout << "camera_info available" << std::endl;

        boost::array<double,9> cam_info = p_cam_info->K;

        K_ = cv::Matx33f(cam_info.at(0), cam_info.at(1), cam_info.at(2),
                         cam_info.at(3), cam_info.at(4), cam_info.at(5),
                         cam_info.at(6), cam_info.at(7), cam_info.at(8));

        got_cam_info = true;
      }
  }
}

void RgbdEvaluatorPreprocessing::calculateHomography()
{
  uint32_t count = 1;
  bool first_image = true;

  std::vector< ImageData >::iterator it;
  std::vector<cv::Point2f> feature_vector_original;
  std::vector<cv::KeyPoint> keypoint_vector_original;

  btTransform transform_original;
  btTransform transform_camx;
  btTransform transform_camx_to_original;

  cv::Mat image_original;
  cv::Mat image_grayscale;

  cv::Matx33f homography_complete;

  std::vector<float> scalings;
  std::vector<float> rotations;
  std::vector<float> angles;

  float angle_orig;
  float dist_orig;
  float rotation_orig;

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
  for (it = it_begin; it != it_end; it+=it_step, count++)
  {
    // write data to file
    std::string fileName;

    // convert integer to string
    std::stringstream ss;
    ss << count;
    std::string count_str = ss.str();

    fileName.append(file_created_folder_);
    fileName.append("/");
    fileName.append("img");
    fileName.append(count_str);

    std::cout << "Writing to "<< fileName << std::endl;

    cv::imwrite(std::string(fileName+".ppm"), it->rgb_image->image );

#if 0
    // Convert 8-Bit to Float image
    cv::Mat1w depth_img;
    it->depth_image->image.convertTo( depth_img, CV_16U, 1000.0, 0.0 );

    std::ofstream fs( (fileName+"_depth.pgm").c_str() );

    fs << "P2" << std::endl;
    fs << depth_img.cols << " " << depth_img.rows << std::endl;
    fs << 65535 << std::endl;

    for ( int y=0; y<depth_img.rows; y++ )
    {
      for ( int x=0; x<depth_img.cols; x++ )
      {
        fs << depth_img[y][x] << " ";
      }
      fs << std::endl;
    }

    std::ofstream mfs( (fileName+"_pose").c_str() );

    btMatrix3x3 basis = it->approx_transform->getBasis();
    btVector3 origin = it->approx_transform->getOrigin();
    for ( int y=0; y<3; y++ )
    {
      for ( int x=0; x<3; x++ )
      {
        mfs << basis[y][x] << " ";
      }
      mfs << std::endl;
    }

    mfs << origin.x() << " " << origin.y() << " " << origin.z() << std::endl;
#endif

    // store first transforms and images
    if(first_image)
    {
      // get original image
      transform_original = *( it->approx_transform.get() );
      image_original = it->rgb_image.get()->image;
      first_image = false;

      btVector3 zvec = transform_original.inverse().getBasis() * btVector3(0,0,1);
      btVector3 xvec = transform_original.inverse().getBasis() * btVector3(1,0,0);

      angle_orig = zvec.angle( btVector3(0,0,-1) ) / M_PI*180.0;
      dist_orig = transform_original.getOrigin().length();
      rotation_orig = xvec.angle( btVector3(1,0,0) ) / M_PI*180.0;

      std::cout << "angle_orig " << angle_orig << std::endl;
      std::cout << "dist_orig " << dist_orig << std::endl;
      std::cout << "rotation_orig " << rotation_orig << std::endl;

      // convert image to grayscale
      cv::cvtColor( image_original, image_grayscale, CV_RGB2GRAY );
      cv::goodFeaturesToTrack( image_grayscale, feature_vector_original, MAX_FEATURE_NUMBER, 0.01,
                               MIN_FEATURE_NEIGHBOUR_DIST, cv::noArray(), 3, true );

      cv::KeyPoint::convert( feature_vector_original, keypoint_vector_original );
#if 0
      cv::drawKeypoints( image_original, keypoint_vector_original, image_original );

       //store original corner points
      cv::imshow("Keypoints", image_original);
      cv::waitKey(30);
#endif
      continue;
    }

    /**************************** calculate initial homography ***************************************************************/

    transform_camx = *(it->approx_transform.get());

    // calculate transform from camera position x to original position
    transform_camx_to_original = transform_original * transform_camx.inverse();

    btVector3 zvec = transform_camx.inverse().getBasis() * btVector3(0,0,1);
    btVector3 xvec = transform_camx.inverse().getBasis() * btVector3(1,0,0);

    float angle_abs = zvec.angle( btVector3(0,0,-1) ) / M_PI*180.0;
    float dist_abs = transform_camx.getOrigin().length();
    float rotation_abs = xvec.angle( btVector3(1,0,0) ) / M_PI*180.0;

    std::cout << "angle " << angle_abs << std::endl;
    std::cout << "dist " << dist_abs << std::endl;
    std::cout << "rotation " << rotation_abs << std::endl;

    float scaling = dist_orig / dist_abs;
    float rotation = std::abs( rotation_abs - rotation_orig );
    float angle = std::abs( angle_abs - angle_orig );

    std::cout << "angle_rel " << angle << std::endl;
    std::cout << "scaling " << scaling << std::endl;
    std::cout << "rotation_rel " << rotation << std::endl;

    angles.push_back( angle );
    rotations.push_back( rotation );
    scalings.push_back( scaling );

    cv::Matx33f homography_init = calculateInitialHomography( transform_camx_to_original, transform_camx );

    cv::Mat image_camx = it->rgb_image.get()->image;
    cv::Mat image_warped;

    // perspective warping
    cv::warpPerspective( image_camx, image_warped, cv::Mat(homography_init), cv::Size(image_original.cols,image_original.rows) );


    cv::Mat tmp1,image_orig_rewarped;
    cv::warpPerspective( image_original, tmp1, cv::Mat(homography_init.inv()), cv::Size(image_original.cols,image_original.rows) );
    cv::warpPerspective( tmp1, image_orig_rewarped, cv::Mat(homography_init), cv::Size(image_original.cols,image_original.rows) );

#if 0
    cv::imshow( "Original Image (Re-Warped)", image_orig_rewarped );
    cv::waitKey(30);
    cv::imshow( "Warped Image", image_warped );
    cv::waitKey(30);
#endif

    /**************************** calculate precise homography **************************************************************/

    std::vector<cv::Point2f> keypoints_camx;
    std::vector<cv::Point2f> keypoints_original;

    for(uint32_t i = 0; i < keypoint_vector_original.size(); i++)
    {
      // ncc part
      cv::Mat result;
      cv::Point2f keypointNCC;

      if((i % 10) == 0)
      {
        //std::cout << "Progress: " << (int)(((float)i/(float)keypoint_vector_original.size())*100) << std::endl;
      }

      if( calculateNCC( image_warped, image_orig_rewarped, keypoint_vector_original.at(i), keypointNCC) >= 0 )
      {
        keypoints_original.push_back( cv::Point2f( keypoint_vector_original.at(i).pt.x,
                                               keypoint_vector_original.at(i).pt.y ) );

        keypoints_camx.push_back( keypointNCC );
      }
    }

    printf("Finished...Found correspondences: %d\n\r", (uint32_t)keypoints_camx.size() );

    if(keypoints_camx.size() < MIN_CORRESPONDENCES || keypoints_original.size() < MIN_CORRESPONDENCES )
    {
      std::cout << "Not enough correspondences found! Exiting..." << std::endl;
      return;
    }

    cv::Matx33f homography_precise;
//    homography_precise = cv::findHomography( keypoints_camx, keypoints_original , CV_LMEDS, 2 );
    homography_precise = cv::findHomography( keypoints_camx, keypoints_original, CV_RANSAC, 2 );

    homography_complete = homography_precise * homography_init;

    printMat( cv::Matx33f( homography_complete) );

    cv::Mat image_warped_precise;

    // perspective warping precise
    cv::warpPerspective( image_camx, image_warped_precise, cv::Mat(homography_complete), cv::Size(image_warped.cols,image_warped.rows) );

    //transform keypoints from warped image into precise warped image
    for ( uint32_t i=0; i < keypoints_camx.size(); i++ )
    {
      cv::Matx31f kp_xyw( keypoints_camx[i].x, keypoints_camx[i].y, 1);

      kp_xyw = homography_precise * kp_xyw;
      //corner_vector_src;
      keypoints_camx[i].x = kp_xyw.val[0] / kp_xyw.val[2];
      keypoints_camx[i].y = kp_xyw.val[1] / kp_xyw.val[2];
    }

    cv::Mat image_original_clone( image_original.clone() );

    // show error via lines
    for ( uint32_t i=0; i<keypoints_camx.size(); i++ )
    {
      cv::line( image_warped_precise, keypoints_camx[i], keypoints_original[i], cv::Scalar(0,0,255), 1 );
      cv::line( image_original_clone, keypoints_camx[i], keypoints_original[i], cv::Scalar(0,0,255), 1 );
    }

    //cv::imshow("Error Lines", image_original_clone );

    /*************************************************************************************************************************/

    // store homography
    writeHomographyToFile( homography_complete.inv(), count );

#if 0

    cv::Mat image_orig_rewarped_precise;
    cv::warpPerspective( image_original, tmp1, cv::Mat(homography_complete.inv()), cv::Size(image_original.cols,image_original.rows) );
    cv::warpPerspective( tmp1, image_orig_rewarped_precise, cv::Mat(homography_complete), cv::Size(image_original.cols,image_original.rows) );
    cv::imshow( "image_orig_rewarped_precise", image_orig_rewarped_precise );

    cv::Mat diff_img;
    cv::absdiff( image_orig_rewarped_precise, image_warped_precise, diff_img );
    cv::imshow( "diff_img", diff_img );
    cv::waitKey(30);
#endif

#if 1
    // show images
    cv::imshow("Current Image", image_camx);
    cv::imshow("Precise Warped Image", image_warped_precise);
    cv::waitKey(30);

    cv::imwrite( file_created_folder_ + "/" + "warped" + count_str + ".ppm", image_warped_precise );

    std::cout << "Press any key to continue" << std::endl;
    //getchar();
#endif
  }

  writeVectorToFile( rotations, "rotation" );
  writeVectorToFile( scalings, "scaling" );
  writeVectorToFile( angles, "viewpoint angle" );
}

cv::Matx33f RgbdEvaluatorPreprocessing::calculateInitialHomography(btTransform transform_camx_to_original, btTransform transform_camx)
{
  // Translation
  tf::Point T_temp = transform_camx_to_original.getOrigin();
  cv::Matx31f T(T_temp.x(), T_temp.y(), T_temp.z());

  //Rotation
  btMatrix3x3 R_temp(transform_camx_to_original.getRotation());
  cv::Matx33f R( R_temp.getColumn(0).getX(), R_temp.getColumn(1).getX(), R_temp.getColumn(2).getX(),
                 R_temp.getColumn(0).getY(), R_temp.getColumn(1).getY(), R_temp.getColumn(2).getY(),
                 R_temp.getColumn(0).getZ(), R_temp.getColumn(1).getZ(), R_temp.getColumn(2).getZ());

  //N
  tf::Vector3 N_temp = transform_camx.getBasis() * btVector3(0,0,1);
  cv::Matx13f N(N_temp.x(), N_temp.y(), N_temp.z());

  //d
  T_temp = transform_camx.getOrigin();
  float_t d = ( N * (cv::Matx31f(T_temp.x(), T_temp.y(), T_temp.z())) ) (0);

  //Calculate init Homography
  cv::Matx33f homography_init = R + (1/d) * T * N;

  // + intrinsic-parameter-matrix
  homography_init = K_ * homography_init * K_.inv();

  return homography_init;
}

int32_t RgbdEvaluatorPreprocessing::calculateNCC(cv::Mat image_original, cv::Mat image_cam_x, cv::KeyPoint keypoint, cv::Point2f& keypointNCC)
{
  float_t x_pos = keypoint.pt.x;
  float_t y_pos = keypoint.pt.y;
  cv::Mat correlation_img;

  if( ( x_pos - round( SEARCH_WINDOW_SIZE / 2 )-1) < 0 ||
      ( y_pos - round( SEARCH_WINDOW_SIZE / 2 )-1) < 0 ||
      ( x_pos + round( SEARCH_WINDOW_SIZE / 2 )+1) > image_cam_x.cols ||
      ( y_pos + round( SEARCH_WINDOW_SIZE / 2 )+1) > image_cam_x.rows )
  {
    //std::cout << "nccSlidingWindow: keypoint( " << x_pos << ", " << y_pos << " ) out of range" << std::endl;
    return -1;
  }

  cv::Rect batch( x_pos - floor( SLIDING_WINDOW_SIZE / 2 ),
                  y_pos - floor( SLIDING_WINDOW_SIZE / 2 ),
                  SLIDING_WINDOW_SIZE, SLIDING_WINDOW_SIZE);

  cv::Rect searchRect( x_pos - floor( SEARCH_WINDOW_SIZE / 2 ),
                  y_pos - floor( SEARCH_WINDOW_SIZE / 2 ),
                  SEARCH_WINDOW_SIZE, SEARCH_WINDOW_SIZE);

  cv::Mat templ( image_cam_x, batch );
  cv::Mat searchWin( image_original, searchRect );

  //cv::imshow("Batch Image", templ);
  //cv::waitKey(30);

  cv::matchTemplate( searchWin, templ, correlation_img, CV_TM_CCORR_NORMED );

  /* find best matches location */
  cv::Point minloc, maxloc;
  double minval = 0, maxval = 0;

  cv::minMaxLoc(correlation_img, &minval, &maxval, &minloc, &maxloc, cv::noArray());

  keypointNCC = keypoint.pt + cv::Point2f(maxloc.x,maxloc.y) -
      cv::Point2f( (SEARCH_WINDOW_SIZE - SLIDING_WINDOW_SIZE) / 2, (SEARCH_WINDOW_SIZE - SLIDING_WINDOW_SIZE) / 2 );

#if 0
  std::cout << "slidingWindow Matrix( " << correlation_img.rows << ", " << correlation_img.cols << " )" << " ... Channels: " << correlation_img.channels()<< std::endl;
  std::cout << "Minval: " << minval << " Maxval: " << maxval << std::endl;
  std::cout << "MinLoc: " << minloc.x <<  "  " << minloc.y <<  " MaxLoc: " << maxloc.x <<  "  " << maxloc.y  << std::endl;
#endif

  if ( maxval < 0.98 )
  {
    return -1;
  }

#if 0
  cv::imshow("correlation_img", correlation_img);

  cv::Rect maxCorrWin( maxloc.x, maxloc.y, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_SIZE);
  cv::Mat maxCorrPatch( searchWin, maxCorrWin );
  cv::imshow("searchWin", searchWin);
  cv::imshow("maxCorrPatch", maxCorrPatch);
  cv::waitKey(30);
  getchar();
#endif

  return 0;
}


void RgbdEvaluatorPreprocessing::writeHomographyToFile(cv::Matx33f homography, uint32_t count)
{
  uint32_t i,j;
  std::fstream file;

  std::stringstream ss;
  ss << count;

  // create filepath
  std::string homographyName;
  homographyName.append(file_created_folder_);
  homographyName.append("/");
  homographyName.append("H1to");
  homographyName.append(ss.str());
  homographyName.append("p");

  file.open(homographyName.c_str(), std::ios::out);

  for(i=0; i<3; i++)
  {
    for(j=0;j<3;j++)
    {
      file << homography(i,j) << "\t";
    }
    file << std::endl;
  }

  file.close();
}

void RgbdEvaluatorPreprocessing::writeVectorToFile( std::vector<float> vec, std::string filename )
{
  uint32_t i,j;
  std::fstream file;

  // create filepath
  std::string homographyName;
  homographyName.append(file_created_folder_);
  homographyName.append("/");
  homographyName.append(filename);

  file.open(homographyName.c_str(), std::ios::out);

  for(i=0; i<vec.size(); i++)
  {
    file << vec[i] << "\t";
  }

  file.close();
}
void RgbdEvaluatorPreprocessing::printMat( cv::Matx33f M )
{
  std::cout << std::setprecision( 3 ) << std::right << std::fixed;
  for ( int row = 0; row < 3; ++ row )
  {
    for ( int col = 0; col < 3; ++ col )
    {
      std::cout << std::setw( 5 ) << (double)M( row, col ) << " ";
    }
    std::cout << std::endl;
  }
}

void RgbdEvaluatorPreprocessing::splitFileName(const std::string& str)
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

} // end namespace


int main( int argc, char** argv )
{
  if(argc < 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << " <bagfileName>" << std::endl;
    return -1;
  }

  std::string fileName(argv[1]);

  bool reverse_order = argc > 2 && std::string(argv[2]) == "-r";

  std::cout << "reverse_order " << reverse_order << std::endl;

  rgbd_evaluator::RgbdEvaluatorPreprocessing fd(fileName, reverse_order);
  fd.createTestFiles();
  fd.calculateHomography();

  std::cout << "Exiting.." << std::endl;
  return 0;
}

