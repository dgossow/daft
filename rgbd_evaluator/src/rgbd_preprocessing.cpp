/*
 * rgbd_evaluator_preprocessing.cpp
 *
 *  Created on: Jan 13, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/rgbd_preprocessing.h"

#include <iostream>
#include <fstream>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <LinearMath/btQuaternion.h>
#include <LinearMath/btMatrix3x3.h>
#include <LinearMath/btVector3.h>

#include <Eigen/Eigenvalues>

namespace rgbd_evaluator
{

RgbdEvaluatorPreprocessing::RgbdEvaluatorPreprocessing(std::string file_path, bool reverse_order, int start_img)
{
  std::cout << "Reading bagfile from " << file_path.c_str() << std::endl;

  reverse_order_ = reverse_order;
  start_img_ = start_img;

  first_image_ = true;
  finishedROI_ = false;
  finishedKP_ = false;

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
  topics.push_back("cam_info");

  rosbag::View view(bag_, rosbag::TopicQuery(topics));

  sensor_msgs::Image::ConstPtr p_current_img;
  geometry_msgs::TransformStamped::ConstPtr p_current_transform;
  sensor_msgs::CameraInfo::ConstPtr p_cam_info;

  ImageData image_data;

  // Load all messages into our stereo dataset
  //BOOST_FOREACH(rosbag::MessageInstance const m, view)
  for ( rosbag::View::iterator it=view.begin(); it!=view.end(); it++ )
  {
      rosbag::MessageInstance& m = *it;

      // load cam_info
      sensor_msgs::CameraInfo::ConstPtr p_cam_info = m.instantiate<sensor_msgs::CameraInfo>();
      if(( p_cam_info != NULL ) && ( got_cam_info == false ))
      {
        //std::cout << "camera_info available" << std::endl;

        boost::array<double,9> cam_info = p_cam_info->K;

        K_ = cv::Matx33f(cam_info.at(0), cam_info.at(1), cam_info.at(2),
                         cam_info.at(3), cam_info.at(4), cam_info.at(5),
                         cam_info.at(6), cam_info.at(7), cam_info.at(8));

        writeIntrinsicMatToFile(K_);

        got_cam_info = true;
      }

      // load rgb image
      sensor_msgs::Image::ConstPtr p_rgb_img = m.instantiate<sensor_msgs::Image>();
      //check if rgb_img message arrived
      if (p_rgb_img != NULL && p_rgb_img->encoding == "bgr8" )
      {
        if ( image_data.rgb_image.rows > 0 )
        {
          std::cout << "There is already an rgb image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_rgb_img);

        // store data in vectorImageData
        image_data.rgb_image = ptr->image;
      }

      // load depth image
      sensor_msgs::Image::ConstPtr p_depth_img = m.instantiate<sensor_msgs::Image>();
      //check if depth_img message arrived
      if (p_depth_img != NULL && p_depth_img->encoding == "32FC1" )
      {
        if ( image_data.depth_image.rows > 0 )
        {
          std::cout << "There is already an depth image for the current dataset! Bagfile invalid." << std::endl;
          return;
        }

        // transform bag image to cvimage
        cv_bridge::CvImagePtr ptr = cv_bridge::toCvCopy(p_depth_img);

        // store data in vectorImageData
        image_data.depth_image = ptr->image;
      }

      // if the current image data is complete, go to next one
      if ( image_data.isComplete() )
      {
        cv::Mat depth_image_orig = image_data.depth_image;
        cv::Mat intensity_image_orig = image_data.rgb_image;

        int scale_fac =   intensity_image_orig.cols / depth_image_orig.cols;

        cv::Mat depth_image, intensity_image;

        // Resize depth to have the same width as rgb
        cv::resize( depth_image_orig, depth_image, cvSize(0,0), scale_fac, scale_fac, cv::INTER_NEAREST );
        // Crop rgb so it has the same size as depth
        intensity_image = cv::Mat( intensity_image_orig, cv::Rect( 0,0, depth_image.cols, depth_image.rows ) );

        image_data.rgb_image = intensity_image;
        image_data.depth_image = depth_image;

        image_store_.push_back( image_data );
        image_data = ImageData();
        std::cout << "Storing dataset #" << count << std::endl;
        count++;
      }
  }
}

void RgbdEvaluatorPreprocessing::writeDepth( cv::Mat& depth_img_orig, std::string count_str )
{
  // write depth map
  // Convert float to 16-bit int
  cv::Mat1w depth_img;
  depth_img_orig.convertTo( depth_img, CV_16U, 1000.0, 0.0 );

  std::ofstream fs( (file_created_folder_ + "/" + "depth" + count_str + ".pgm").c_str() );

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
}

float RgbdEvaluatorPreprocessing::getAspect( std::vector< cv::Point2f > pts )
{
  cv::Point2f minp(100000,100000),maxp(0,0);
  uint32_t k;
  for(k = 0; k < pts.size(); k++)
  {
    maxp.x = std::max ( pts[k].x, maxp.x );
    maxp.y = std::max ( pts[k].y, maxp.y );
    minp.x = std::min ( pts[k].x, minp.x );
    minp.y = std::min ( pts[k].y, minp.y );
  }

  return (maxp.x/minp.x) / (maxp.y/minp.y);
}

void RgbdEvaluatorPreprocessing::calculateHomography()
{
  uint32_t count = 1;

  std::vector< ImageData >::iterator it;
  std::vector<cv::Point2f> feature_vector_img1;
  std::vector<cv::KeyPoint> kp_vec_img1;
  std::vector<cv::Point2f> maskPoints_vector;

  cv::Mat img1;
  cv::Mat last_imgx;

  cv::Matx33f homography_final;
  cv::Matx33f homography_complete_last = cv::Matx33f::eye();

  std::vector<float> scalings;
  std::vector<float> rotations;
  std::vector<float> angles;

  btVector3 xvec_orig;
  btVector3 zvec_orig;
  float dist_orig = 0.0;
  float aspect_orig = 0.0;

  tf::StampedTransform transform_original;

  int it_step;
  std::vector< ImageData >::iterator it_end,it_begin;
  if ( reverse_order_ )
  {
    it_step = -1;
    it_begin = image_store_.end()-1-start_img_;
    it_end = image_store_.begin();
  }
  else
  {
    it_step = 1;
    it_begin = image_store_.begin()+start_img_;
    it_end = image_store_.end();
  }

  for (it = it_begin; it != it_end; it+=it_step, count++)
  {
    // convert integer to string
    std::stringstream ss;
    ss << count;
    std::string count_str = ss.str();

    // get image from storage
    cv::Mat imgx = it->rgb_image;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    std::string windowNameROI = "Choose Area of interest";

    if(!finishedROI_)
    {
      // image for choosing polygon with region of interest
      imageChooseROI_ = imgx.clone();

      cv::imshow( windowNameROI , imageChooseROI_ );
      cv::waitKey(50);

      // Set up the callback for choosing polygon
      cv::setMouseCallback( windowNameROI, imgMouseCallbackROI, this);
    }

    std::vector<cv::KeyPoint> tmpPointsROI;
    cv::Point2f lastPoint;

    // wait for mouse events
    while( !finishedROI_ )
    {
      // got at least two points
      if( this->mousePointsROI_.size() > 1 )
      {
        // connect lines
        //cv::polylines(imageChooseROI_, mousePointsROI_, true, CV_RGB(0, 0, 255) );
        cv::line(imageChooseROI_, lastPoint, this->mousePointsROI_.back(), CV_RGB(0, 255, 0));
      }

      if( this->mousePointsROI_.size() > 0 )
      {
        cv::KeyPoint::convert( mousePointsROI_, tmpPointsROI );
        cv::drawKeypoints( imageChooseROI_, tmpPointsROI, imageChooseROI_, CV_RGB(0, 255, 0) );
        cv::imshow( windowNameROI, imageChooseROI_ );

        // store last point
        lastPoint = this->mousePointsROI_.back();
      }

      cv::waitKey(50);
    }

    cv::Mat1b mask_img;
    if ( mousePointsROI_.size() < 3 )
    {
      mask_img = cv::Mat::ones(imageChooseROI_.rows, imageChooseROI_.cols, CV_8U);
    }
    else
    {
      // connect first and last point
      cv::line(imageChooseROI_, this->mousePointsROI_.at(0), this->mousePointsROI_.back(), CV_RGB(0, 255, 0));

      // fill polynom
      std::vector<cv::Point> pts;

      for(uint32_t k = 0; k < mousePointsROI_.size(); k++)
      {
        pts.push_back( mousePointsROI_.at(k) );
      }

      cv::Point *points;
      points = &pts[0];
      int nbtab = pts.size();

      mask_img = cv::Mat1b::zeros(imageChooseROI_.rows, imageChooseROI_.cols);
      cv::fillPoly(mask_img, (const cv::Point **) &points, &nbtab, 1, CV_RGB(255,255,255));
    }

    cv::imshow("Mask", mask_img);

    cv::KeyPoint::convert( mousePointsROI_, tmpPointsROI );
    cv::drawKeypoints( imageChooseROI_, tmpPointsROI, imageChooseROI_, CV_RGB(0, 255, 0) );
    cv::imshow( windowNameROI, imageChooseROI_ );

    std::string maskf = file_created_folder_ + "/" + "mask.pgm";
    std::cout << "Writing " << maskf << std::endl;
    cv::imwrite(maskf, mask_img);

    cv::waitKey(50);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // image for choosing and drawing keypoints
    keyPointImageOrigin_ = imgx.clone();

    std::string windowName = "ImageX: Mark Keypoints. Right click to finish!";

    // for mouse callback
    cv::namedWindow( windowName );

    // Set up the callback
    cv::setMouseCallback( windowName, imgMouseCallbackKP, this);

    for(int h = 0; h < keyPointImageOrigin_.cols; h++)
    {
      for(int t = 0; t < keyPointImageOrigin_.rows; t++)
      {
          if (std::isnan(it->depth_image.at<float>(t,h)))
          {
            keyPointImageOrigin_.col(h).row(t) = cv::Scalar(0,(((h+t)/2)%2),0);
          }
      }
    }

    cv::imshow( windowName, keyPointImageOrigin_ );

    // store first transforms and images
    if(first_image_)
    {
      cv::imshow("Image Original", keyPointImageOrigin_);

      // get original image
      img1 = imgx;
      last_imgx = img1;

      std::vector<cv::KeyPoint> tmpKeypoints;

      // wait for keypoints in the first image and draw them
      while( (this->mouseKeypointsOrigin_.size() < MIN_CORRESPONDENCES) || finishedKP_ == false )
      {
        if(this->mouseKeypointsOrigin_.size() > 0 )
        {
          cv::KeyPoint::convert( mouseKeypointsOrigin_, tmpKeypoints );
          cv::drawKeypoints(keyPointImageOrigin_, tmpKeypoints, keyPointImageOrigin_, CV_RGB(255,0,0));
          cv::imshow(windowName, keyPointImageOrigin_);
          cv::imshow("Image Original", keyPointImageOrigin_);
        }
        cv::waitKey(50);
      }

      // reset flag
      finishedKP_ = false;

      cv::KeyPoint::convert( mouseKeypointsOrigin_, tmpKeypoints );
      cv::drawKeypoints(keyPointImageOrigin_, tmpKeypoints, keyPointImageOrigin_, CV_RGB(255,0,0));
      cv::imshow(windowName, keyPointImageOrigin_);
      cv::imshow("Image Original", keyPointImageOrigin_);
      cv::waitKey(50);
      // stop drawing

      uint32_t k;
      for(k = 0; k < mouseKeypointsOrigin_.size(); k++)
      {
        maskPoints_vector.push_back(mouseKeypointsOrigin_.at(k));
      }

      // first image done, received at least 4 points
      std::cout << "First Image processed!" << std::endl;

      // Calculate angle_orig, dist_orig, rotation_orig
      transform_original = calculateCoordinatesystem(it->depth_image, mouseKeypointsOrigin_);

      aspect_orig = getAspect( mouseKeypointsOrigin_ );

      zvec_orig = transform_original.getBasis() * btVector3(0,0,1);
      xvec_orig = transform_original.getBasis() * btVector3(1,0,0);
      dist_orig = transform_original.getOrigin().length();

      std::cout << "aspect_orig " << aspect_orig << std::endl;
      std::cout << "angle_orig " << zvec_orig.angle( btVector3(0,0,-1) ) / M_PI*180.0 << std::endl;
      std::cout << "dist_orig " << dist_orig << std::endl;

      // convert image to grayscale
      cv::Mat image_grayscale;
      cv::cvtColor( img1, image_grayscale, CV_RGB2GRAY );

      /*
      cv::goodFeaturesToTrack( image_grayscale, feature_vector_img1, MAX_FEATURE_NUMBER, 0.01,
                               MIN_FEATURE_NEIGHBOUR_DIST, cv::noArray(), 3, true );
      cv::KeyPoint::convert( feature_vector_img1, kp_vec_img1 );\
      */

      // save mouse clicks as keypoints
      kp_vec_img1.clear();
      for ( uint32_t i=0; i<mouseKeypointsOrigin_.size(); i++ )
      {
        kp_vec_img1.push_back( cv::KeyPoint( mouseKeypointsOrigin_[i], 1 ) );
      }

      cv::imwrite( file_created_folder_ + "/" + "img1.ppm", img1 );

      first_image_ = false;

    } // endif first img
    else
    {
      /**************************** calculate initial homography ***************************************************************/

      std::vector<cv::KeyPoint> tmpKeypoints;

      // wait for keypoints in image x
      while( this->mouseKeypointsImageX_.size() < MIN_CORRESPONDENCES || finishedKP_ == false )
      {
       if(this->mouseKeypointsImageX_.size() > 0 )
       {
         cv::KeyPoint::convert( mouseKeypointsImageX_, tmpKeypoints );
         cv::drawKeypoints(keyPointImageOrigin_, tmpKeypoints, keyPointImageOrigin_,CV_RGB(255,0,0));
         cv::imshow(windowName, keyPointImageOrigin_);
       }
       cv::waitKey(50);
      }
      // reset flag
      finishedKP_ = false;

      cv::KeyPoint::convert( mouseKeypointsImageX_, tmpKeypoints );
      cv::drawKeypoints(keyPointImageOrigin_, tmpKeypoints, keyPointImageOrigin_,CV_RGB(255,0,0));
      cv::imshow(windowName, keyPointImageOrigin_);
      cv::waitKey(50);

      // store keypoints to write to file
      uint32_t k;
      for(k = 0; k < mouseKeypointsImageX_.size(); k++)
      {
        maskPoints_vector.push_back(mouseKeypointsImageX_.at(k));
      }

      std::cout << std::endl;
      std::cout << "Image_" << count_str << " processed!" << std::endl;

      tf::StampedTransform transform_camx;
      transform_camx = calculateCoordinatesystem(it->depth_image, mouseKeypointsImageX_);

      // calculate transform from camera position x to original position
      tf::StampedTransform transform_camx_to_original;
      transform_camx_to_original.mult(transform_camx.inverse(),transform_original);

      btVector3 zvec = transform_camx.getBasis() * btVector3(0,0,1);
      btVector3 xvec = transform_camx.getBasis() * btVector3(1,0,0);

      float dist_abs = transform_camx.getOrigin().length();
      std::cout << "dist " << dist_abs << std::endl;

      float scaling = dist_orig / dist_abs;
      float rotation = xvec.angle( xvec_orig ) / M_PI*180.0;
      float angle = zvec.angle( zvec_orig ) / M_PI*180.0;

      std::cout << "angle_rel " << angle << std::endl;
      std::cout << "scaling " << scaling << std::endl;
      std::cout << "rotation_rel " << rotation << std::endl;

      angles.push_back( angle );
      rotations.push_back( rotation );
      scalings.push_back( scaling );

      if(mouseKeypointsImageX_.size() != mouseKeypointsOrigin_.size())
      {
        std::cout << "Number of Keypoints does not match. Unable to calculate Homography!" << std::endl;
        return;
      }

      cv::Matx33f homography_approx = cv::findHomography( mouseKeypointsImageX_, mouseKeypointsOrigin_, CV_RANSAC );

      // warp images with approx. homography
      cv::Mat imgx_warped_approx;
      cv::warpPerspective( imgx, imgx_warped_approx, cv::Mat(homography_approx), cv::Size( img1.cols, img1.rows ) );

      cv::Mat tmp1,img1_rewarped;
      cv::warpPerspective( img1, tmp1, cv::Mat( homography_approx.inv() ), cv::Size( img1.cols, img1.rows ) );
      cv::warpPerspective( tmp1, img1_rewarped, cv::Mat(homography_approx), cv::Size( img1.cols,img1.rows ) );

#if 1
      cv::imshow( "Warped Image approx", imgx_warped_approx );
      cv::waitKey(30);
#endif

      /**************************** calculate precise homography **************************************************************/

      std::vector<cv::Point2f> kp_pts_imgx;
      std::vector<cv::Point2f> kp_pts_img1;

      for(uint32_t i = 0; i < kp_vec_img1.size(); i++)
      {
        // ncc part
        cv::Mat result;
        cv::Point2f keypointNCC;

        if( calculateNCC( imgx_warped_approx, img1_rewarped, kp_vec_img1.at(i), keypointNCC, i ) >= 0 )
        {
          kp_pts_img1.push_back( cv::Point2f( kp_vec_img1.at(i).pt.x,
                                              kp_vec_img1.at(i).pt.y ) );

          kp_pts_imgx.push_back( keypointNCC );
        }
      }

      // check amount of correspondences
      if(kp_pts_imgx.size() < MIN_CORRESPONDENCES || kp_pts_img1.size() < MIN_CORRESPONDENCES )
      {
        std::cout << "Not enough correspondences found! Exiting..." << std::endl;
        return;
      }

      cv::Matx33f homography_precise;

      homography_precise = cv::findHomography( kp_pts_imgx, kp_pts_img1, CV_RANSAC, 2 );

      homography_final = homography_precise * homography_approx;
      homography_final *= 1.0 / homography_final(2,2);
      homography_complete_last = homography_final;

      cv::Mat imgx_warped_final;

      // perspective warping precise
      cv::warpPerspective( imgx, imgx_warped_final, cv::Mat(homography_final), cv::Size(imgx_warped_approx.cols,imgx_warped_approx.rows) );

      // normalize lightness
      cv::Scalar mean_img1_rewarped = cv::mean( img1_rewarped );
      cv::Scalar mean_imgx_warped_final = cv::mean( imgx_warped_final );

      float l_img1_rewarped = (mean_img1_rewarped[0] + mean_img1_rewarped[1] + mean_img1_rewarped[2]) / 3.0;
      float l_imgx_warped_final = (mean_imgx_warped_final[0] + mean_imgx_warped_final[1] + mean_imgx_warped_final[2]) / 3.0;

      imgx *= l_img1_rewarped / l_imgx_warped_final;
      imgx_warped_final *= l_img1_rewarped / l_imgx_warped_final;

      // transform keypoints from warped image into precise warped image
      for ( uint32_t i=0; i < kp_pts_imgx.size(); i++ )
      {
        cv::Matx31f kp_xyw( kp_pts_imgx[i].x, kp_pts_imgx[i].y, 1);

        kp_xyw = homography_precise * kp_xyw;
        //corner_vector_src;
        kp_pts_imgx[i].x = kp_xyw.val[0] / kp_xyw.val[2];
        kp_pts_imgx[i].y = kp_xyw.val[1] / kp_xyw.val[2];
      }

      cv::Mat image_original_clone( img1.clone() );

#if 0
      // show error via lines
      for ( uint32_t i=0; i<kp_pts_imgx.size(); i++ )
      {
        cv::line( imgx_warped_final, kp_pts_imgx[i], kp_pts_img1[i], cv::Scalar(0,0,255), 1 );
        cv::line( image_original_clone, kp_pts_imgx[i], kp_pts_img1[i], cv::Scalar(0,0,255), 1 );
      }
#endif
      /*************************************************************************************************************************/

      // store homography
      writeHomographyToFile( homography_final.inv(), count );

#if 0

      cv::Mat image_orig_rewarped_precise;
      cv::warpPerspective( img1, tmp1, cv::Mat(homography_final.inv()), cv::Size(img1.cols,img1.rows) );
      cv::warpPerspective( tmp1, image_orig_rewarped_precise, cv::Mat(homography_final), cv::Size(img1.cols,img1.rows) );
      cv::imshow( "image_orig_rewarped_precise", image_orig_rewarped_precise );

      cv::Mat diff_img;
      cv::absdiff( image_orig_rewarped_precise, imgx_warped_final, diff_img );
      cv::imshow( "diff_img", diff_img );
      cv::waitKey(30);
#endif

#if 1
      // show images
      cv::imshow("Final Warped Image", imgx_warped_final);
      cv::waitKey(100);
      cv::imwrite( file_created_folder_ + "/" + "warped" + count_str + ".ppm", imgx_warped_final );

    } // end else

    cv::imwrite( file_created_folder_ + "/" + "img" + count_str + ".ppm", imgx );
    //cv::imshow("depth",it->depth_image->image);
    writeDepth( it->depth_image, count_str );

#if 0
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

    // reset keypoints
    mouseKeypointsImageX_.clear();

#endif
  } // end for

  writeMaskPointsToFile(maskPoints_vector);

  writeVectorToFile( rotations, "rotation" );
  writeVectorToFile( scalings, "scaling" );
  writeVectorToFile( angles, "viewpoint angle" );
}

cv::Matx33f RgbdEvaluatorPreprocessing::calculateInitialHomography( cv::Mat& img1c, cv::Mat& img2c )
{
  cv::Mat img1,img2;
  cv::cvtColor( img1c, img1, CV_BGR2GRAY );
  cv::cvtColor( img2c, img2, CV_BGR2GRAY );

  cv::SIFT::CommonParams cp;
  cv::SIFT::DetectorParams detp;
  cv::SIFT::DescriptorParams descp;
  detp.threshold = detp.GET_DEFAULT_THRESHOLD() * 3;

  cv::SIFT sift( cp, detp, descp );
  std::vector<cv::KeyPoint> kp1,kp2;
  cv::Mat desc1,desc2;
  cv::Mat mask;
  sift( img1, mask, kp1, desc1 );
  sift( img2, mask, kp2, desc2 );

  cv::BruteForceMatcher< cv::L2<float> > matcher;
  cv::vector< cv::DMatch > matches1,matches;
  matcher.match( desc1, desc2, matches1 );

  for ( cv::vector< cv::DMatch >::iterator i=matches1.begin(); i!=matches1.end(); i++ )
  {
    if ( i->distance < 200 )
    {
      matches.push_back( *i );
    }
  }

  cv::Mat disp_img;
  cv::drawMatches( img1, kp1, img2, kp2, matches, disp_img );
  cv::imshow("matches",disp_img);

  cv::waitKey(100);

  std::vector<cv::Point2f> src_pts;
  std::vector<cv::Point2f> dst_pts;

  for ( unsigned i=0; i<matches.size(); i++ )
  {
    int i1 = matches[i].queryIdx;
    int i2 = matches[i].trainIdx;
    src_pts.push_back( kp1[ i1 ].pt );
    dst_pts.push_back( kp2[ i2 ].pt );
  }

  cv::Mat1d hom;
  hom = cv::findHomography( src_pts, dst_pts, CV_RANSAC );

  return hom;
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

int32_t RgbdEvaluatorPreprocessing::calculateNCC(cv::Mat image_original, cv::Mat image_cam_x, cv::KeyPoint keypoint, cv::Point2f& keypointNCC, int i)
{
  float_t x_pos = keypoint.pt.x;
  float_t y_pos = keypoint.pt.y;
  cv::Mat correlation_img;

  if( ( x_pos - round( SEARCH_WINDOW_SIZE / 2 )-1) < 0 ||
      ( y_pos - round( SEARCH_WINDOW_SIZE / 2 )-1) < 0 ||
      ( x_pos + round( SEARCH_WINDOW_SIZE / 2 )+1) > image_cam_x.cols ||
      ( y_pos + round( SEARCH_WINDOW_SIZE / 2 )+1) > image_cam_x.rows )
  {
    std::cout << "Error: control point outside of image boundaries" << std::endl;
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

  cv::matchTemplate( searchWin, templ, correlation_img, CV_TM_CCORR_NORMED );

  /* find best matches location */
  cv::Point minloc, maxloc;
  double minval = 0, maxval = 0;

  cv::minMaxLoc(correlation_img, &minval, &maxval, &minloc, &maxloc, cv::noArray());

  keypointNCC = keypoint.pt + cv::Point2f(maxloc.x,maxloc.y) -
      cv::Point2f( (SEARCH_WINDOW_SIZE - SLIDING_WINDOW_SIZE) / 2, (SEARCH_WINDOW_SIZE - SLIDING_WINDOW_SIZE) / 2 );

#if 1
  std::cout << "slidingWindow Matrix( " << correlation_img.rows << ", " << correlation_img.cols << " )" << " ... Channels: " << correlation_img.channels()<< std::endl;
  std::cout << "Minval: " << minval << " Maxval: " << maxval << std::endl;
  std::cout << "MinLoc: " << minloc.x <<  "  " << minloc.y <<  " MaxLoc: " << maxloc.x <<  "  " << maxloc.y  << std::endl;
#endif

  /*
  if ( maxval < NCC_MAX_VAL )
  {
    std::cout << "Error: control point outside of image boundaries" << std::endl;
    return -1;
  }
  */

#if 1
  std::stringstream s;
  s << i;
  cv::imshow("correlation_img"+s.str(), correlation_img);

  cv::Rect maxCorrWin( maxloc.x, maxloc.y, SLIDING_WINDOW_SIZE, SLIDING_WINDOW_SIZE);
  cv::Mat maxCorrPatch( searchWin, maxCorrWin );
  cv::imshow("searchWin"+s.str(), searchWin);
  cv::imshow("maxCorrPatch"+s.str(), maxCorrPatch);
  cv::waitKey(30);
  //getchar();
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

void RgbdEvaluatorPreprocessing::writeIntrinsicMatToFile(cv::Matx33f K)
{
  uint32_t i,j;
  std::fstream file;

  // create filepath
  std::string intrinsicMatName;
  intrinsicMatName.append(file_created_folder_);
  intrinsicMatName.append("/");
  intrinsicMatName.append("K_");

  file.open(intrinsicMatName.c_str(), std::ios::out);

  for(i=0; i<3; i++)
  {
    for(j=0;j<3;j++)
    {
      file << K(i,j) << "\t";
    }
    file << std::endl;
  }

  file.close();
}

void RgbdEvaluatorPreprocessing::writeVectorToFile( std::vector<float> vec, std::string filename )
{
  uint32_t i;
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

void RgbdEvaluatorPreprocessing::writeMaskPointsToFile( std::vector<cv::Point2f> maskPoints )
{
  uint32_t i = 0;

  std::fstream file;
  std::string fileName;

  fileName.append(file_created_folder_);
  fileName.append("/");
  fileName.append("MaskPoints");

  file.open(fileName.c_str(), std::ios::out);

  for(i = 0; i < maskPoints.size(); i++ )
  {
    file << maskPoints.at(i).x << "  " << maskPoints.at(i).y << std::endl;
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

  /*
  if ( reverse_order_ )
  {
    file_created_folder_.append("_reverse");
  }
  */

  std::cout << " path: " << file_path_ << std::endl;
  std::cout << " file: " << file_name_ << std::endl;
  std::cout << " folder: " << file_folder_ << std::endl;
  std::cout << " created folder: " << file_created_folder_ << std::endl;
}

btVector3 getPt3D( int u, int v, float z, float f_inv, float cx, float cy )
{
  float zf = z*f_inv;
  btVector3 p;
  p[0] = zf * (u-cx);
  p[1] = zf * (v-cy);
  p[2] = z;
  return p;
}

tf::StampedTransform RgbdEvaluatorPreprocessing::calculateCoordinatesystem( cv::Mat& depth_img,  std::vector<cv::Point2f> mouseKeypoints)
{
  float f_inv = 1.0 / K_(0,0);
  float cx  = K_(0,2);
  float cy  = K_(1,2);

  tf::StampedTransform transform_original;
  std::vector<btVector3> CooPoint;
  btVector3 center;

  for(uint32_t i = 0; i < mouseKeypoints.size(); i++)
  {
    float num_zval = 0;
    float z_sum=0;
    for ( int y=-10;y<10;y++ )
    {
      for ( int x=-10;x<10;x++ )
      {
        float z = depth_img.at<float>( mouseKeypoints.at(i).y+y,mouseKeypoints.at(i).x+x );
        if ( !isnan(z) )
        {
          z_sum+=z;
          num_zval++;
        }
      }
    }

    if (num_zval == 0)
    {
      std::cout << "no depth value available!!!" << std::endl;
      exit(0);
    }

    float z = z_sum / num_zval;

    btVector3 CooPoint_tmp = getPt3D(
        mouseKeypoints.at(i).x,
        mouseKeypoints.at(i).y,
        z, f_inv, cx, cy );
    CooPoint.push_back( CooPoint_tmp );

//    std::cout << "Tiefe " << i << ": "  << CooPoint_tmp[2] << std::endl;
    center += CooPoint_tmp;
  }

  center /= float(mouseKeypoints.size());

  btVector3 u = CooPoint[1] - CooPoint[0];
  btVector3 v = CooPoint[2] - CooPoint[0];
  btVector3 w = u.cross(v);
  btVector3 v1 = w.cross( u );

  btMatrix3x3 basis;
  basis[0] = u.normalize();
  basis[1] = v1.normalize();
  basis[2] = w.normalize();
  basis=basis.transpose();

  transform_original.setOrigin( center );
  transform_original.setBasis( basis );

  //std::cout << transform_original.getOrigin().getX() << " " << transform_original.getOrigin().getY() << " " << transform_original.getOrigin().getZ() << std::endl;

  return transform_original;

}

// Implement mouse callback
void RgbdEvaluatorPreprocessing::imgMouseCallbackKP( int event, int x, int y, int flags, void* param )
{
    switch( event )
    {
      case  CV_EVENT_LBUTTONDOWN:

        if(((RgbdEvaluatorPreprocessing*) param)->first_image_)
        {
          ((RgbdEvaluatorPreprocessing*) param)->mouseKeypointsOrigin_.push_back(cv::Point2f(x,y));
          //std::cout << "Origin: Received mouse click on x: " << x << "  y: " << y << std::endl;
        }
        else
        {
          ((RgbdEvaluatorPreprocessing*) param)->mouseKeypointsImageX_.push_back(cv::Point2f(x,y));
          //std::cout << "Camx: Received mouse click on x: " << x << "  y: " << y << std::endl;
        }
        break;
      case CV_EVENT_RBUTTONDOWN:
        ((RgbdEvaluatorPreprocessing*) param)->finishedKP_ = true;
        break;

      default:
        break;
    }
}

void RgbdEvaluatorPreprocessing::imgMouseCallbackROI( int event, int x, int y, int flags, void* param )
{
  if ( !param )
  {
    std::cout << "ERROR: NULL pointer received!" << std::endl;
    throw;
  }

  RgbdEvaluatorPreprocessing* rgb_processing = static_cast<RgbdEvaluatorPreprocessing*>( param );

  switch( event )
  {
    case  CV_EVENT_LBUTTONDOWN:
      rgb_processing->mousePointsROI_.push_back(cv::Point2f(x,y));
      break;

    case  CV_EVENT_RBUTTONDOWN:
      rgb_processing->finishedROI_ = true;
      break;

    default:
      break;
  }
}

} // end namespace


int main( int argc, char** argv )
{
  if(argc < 2)
  {
    std::cout << "Wrong usage, Enter: " << argv[0] << "[-r] <bagfileName> .." << std::endl;
    return -1;
  }

  bool reverse_order = false;
  int start_img = 0;

  int start_i = reverse_order ? 2 : 1;

  for ( int c=1; c<argc-1; c++ )
  {
    std::string arg(argv[c]);
    if ( arg == "-r" )
    {
      reverse_order = true;
      start_i++;
    }
    else if ( arg == "-s" )
    {
      start_img = atoi(argv[c+1]) -1;
      start_i+=2;
    }
  }

  std::cout << "reverse_order " << reverse_order << std::endl;
  std::cout << "start_img " << start_img << std::endl;

  for ( int i=start_i; i<argc; i++ )
  {
    std::string file_name(argv[i]);
    rgbd_evaluator::RgbdEvaluatorPreprocessing fd(file_name, reverse_order, start_img);
    fd.createTestFiles();
    fd.calculateHomography();
  }

  std::cout << "Exiting.." << std::endl;
  return 0;
}

