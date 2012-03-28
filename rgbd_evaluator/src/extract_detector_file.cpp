/*
 * extract_detector_file.cpp
 *
 *  Created on: Feb 20, 2012
 *      Author: praktikum
 */

#include "rgbd_evaluator/extract_detector_file.h"
#include "sift/Sift.h"
#include "parallelsurf/KeyPointDetector.h"
#include "parallelsurf/KeyPointDescriptor.h"
#include "parallelsurf/Image.h"

#include <iostream>
#include <fstream>

#include <math.h>
#include <boost/timer.hpp>

#include <sensor_msgs/image_encodings.h>

#define daft_ns cv::daft2

namespace rgbd_evaluator
{

const int num_kp = 500;
int img_count = 0;

ExtractDetectorFile::ExtractDetectorFile(std::string file_path)
{
  std::cout << "Starting extract_detector_file..." << std::endl;

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

  //system( "rm " );

  extra_folder_ = file_created_folder_ + "/extra";

  if( system(("mkdir "+ extra_folder_).c_str()) < 0) // -1 on error
  {
    std::cout << "Error when executing: " << "mkdir "+ extra_folder_  << std::endl;
    std::cout << "--> check user permissions"  << std::endl;
    return;
  }

  readDataFiles();
  extractAllKeypoints();
}

ExtractDetectorFile::~ExtractDetectorFile()
{
  std::cout << "Stopping extract_detector_file..." << std::endl;
  cv::destroyAllWindows();
}

void ExtractDetectorFile::readDataFiles()
{
  uint32_t i = 0;
  uint32_t numberOfImages = 12;

  std::string path;
  path.append(file_created_folder_);
  path.append("/");
  path.append("K_");

  // read intrinsic matrix K once
  if( !readMatrix( path, K_ ) )
  {
    std::cout << path << " not found - aborting..." << std::endl;
    return;
  }

  std::string maskImagePath;
  maskImagePath.append(file_created_folder_);
  maskImagePath.append("/");
  maskImagePath.append("maskImage.ppm");

  // read mask image
  if ( !fileExists( maskImagePath ) )
  {
    std::cout << maskImagePath << " not found - aborting..." << std::endl;
    return;
  }

  cv::Mat mask_tmp = cv::imread(maskImagePath);
  //mask_tmp.convertTo(maskImage_,CV_BGR2GRAY);

  maskImage_ = mask_tmp;
  std::cout << maskImage_.type() << std::endl;

  // create image paths
  std::string image_rgb_name;
  image_rgb_name.append( file_created_folder_ );
  image_rgb_name.append( "/" );
  image_rgb_name.append( "img" );

  std::string image_depth_name;
  image_depth_name.append( file_created_folder_ );
  image_depth_name.append( "/" );
  image_depth_name.append( "depth" );

  // read rgb and depth images
  for( i = 0; i < numberOfImages; i++ )
  {
    std::cout << "Processing image: " << i+1 << std::endl;

    std::stringstream ss;
    ss << i+1;

    std::string tmp_rgb_name;
    tmp_rgb_name.append( image_rgb_name );
    tmp_rgb_name.append( ss.str() );
    tmp_rgb_name.append( ".ppm" );

    std::string tmp_depth_name;
    tmp_depth_name.append( image_depth_name );
    tmp_depth_name.append( ss.str() );
    tmp_depth_name.append( ".pgm" );

    if( !fileExists( tmp_rgb_name ) )
    {
      std::cout << tmp_rgb_name << " not found! - aborting..." << std::endl;
      return;
    }

    if( !fileExists( tmp_depth_name ) )
    {
      std::cout << tmp_depth_name << " not found! - aborting..." << std::endl;
      return;
    }

    ImageData img_data;

    // read homography
    if ( i==0 )
    {
      img_data.hom = cv::Matx33f::eye();
    }
    else if( !readMatrix( file_created_folder_+"/H1to"+ss.str()+"p", img_data.hom ) )
    {
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

    image_store_.push_back( img_data );
  }
}

bool ExtractDetectorFile::fileExists( const std::string & fileName )
{
  std::ifstream fileTest(fileName.c_str());

  if(!fileTest) return false;

  fileTest.close();
  return true;
}

bool ExtractDetectorFile::readMatrix( const std::string & fileName, cv::Matx33f& K )
{
  static const uint32_t MATRIX_DIM = 3;

  // check if file exists
  if( !fileExists( fileName ) )
  {
    std::cout << "ERROR: " << fileName << " not found!" << std::endl;
    return false;
  }

  // start reading data
  std::ifstream infile( fileName.c_str() );

  K = cv::Matx33f( MATRIX_DIM, MATRIX_DIM );

  for( uint32_t y = 0; y < MATRIX_DIM; y++ )
  {
    for ( uint32_t x=0; x < MATRIX_DIM; x++ )
    {
      if (infile.eof())
      {
        std::cout << "ERROR: end-of-file reached too early!" << std::endl;
        exit(-1);
      }
      float n;
      infile >> n;
      // write values to matrix
      K( y, x ) = n;
    }
  }

  infile.close();
  return true;
}

bool ExtractDetectorFile::readDepth( const std::string & fileName, cv::Mat1f& depth_img )
{
  uint32_t depth_rows = 0, depth_cols = 0;

  std::string input_string;
  std::ifstream infile;

  infile.open( fileName.c_str() );
  getline(infile, input_string); // Header1

  if( input_string != "P2" )
  {
    std::cout << fileName << ": Wrong image Header ( " << input_string << " ) ..." << std::endl;
    return false;
  }

  infile >> depth_cols;
  infile >> depth_rows;

  int maxval;
  infile >> maxval;

  //std::cout << "depth_cols: " << depth_cols << "   depth_rows: " << depth_rows << "   maxval: " << maxval << std::endl;

  depth_img = cv::Mat1f(depth_rows, depth_cols);

  for( uint32_t y = 0; y < depth_rows; y++ )
  {
    for ( uint32_t x=0; x<depth_cols; x++ )
    {
      int n;
      if (infile.eof())
      {
        std::cout << "ERROR: end-of-file reached too early!" << std::endl;
        exit(-1);
      }
      infile >> n;
      if ( n == 0 )
      {
        depth_img( y, x ) = std::numeric_limits<float>::quiet_NaN();
      }
      else
      {
        depth_img( y, x ) = float(n) * 0.001;
      }
    }
  }

  infile.close();

  return true;
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

std::vector<cv::KeyPoint3D> makeKp3d( std::vector<cv::KeyPoint> kp, cv::Mat1d descriptors )
{
  std::vector<cv::KeyPoint3D> kp3d_vec;
  kp3d_vec.reserve( kp.size() );
  for ( size_t k=0; k < kp.size(); k++ )
  {
    cv::KeyPoint3D kp3d = kp[k];
    for ( int i=0; i<descriptors.cols; i++ )
    {
      kp3d.desc.push_back( descriptors[k][i] );
    }
    kp3d_vec.push_back( kp3d );
  }
  return kp3d_vec;
}

// these helper function compute the number of keypoints
// for a given threshold

//define insertor class which collects keypoints in a vector
class VecIns : public parallelsurf::KeyPointInsertor
{
  public:
    VecIns ( std::vector<parallelsurf::KeyPoint>& keyPoints ) : m_KeyPoints ( keyPoints ) {};
    inline virtual void operator() ( const parallelsurf::KeyPoint &keyPoint )
    {
      m_KeyPoints.push_back ( keyPoint );
    }
  private:
    std::vector<parallelsurf::KeyPoint>& m_KeyPoints;
};

std::vector<cv::KeyPoint3D> getSurfKp( const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  unsigned char** pixels = new unsigned char*[gray_img.rows];
  for ( int y=0; y<gray_img.rows; y++ )
  {
    pixels[y] = new unsigned char[gray_img.cols];
    for ( int x=0; x<gray_img.cols; x++ )
    {
      pixels[y][x] = gray_img.at<uchar>(y,x);
    }
  }
  parallelsurf::Image intImage ( (const unsigned char**)pixels, gray_img.cols, gray_img.rows );

  parallelsurf::KeyPointDetector detector;
  detector.setScoreThreshold( 0.1 * t );

  parallelsurf::KeyPointDescriptor descriptor ( intImage, false );
  std::vector<parallelsurf::KeyPoint> surf_kps;
  VecIns insertor( surf_kps );

  detector.detectKeyPoints( intImage, insertor );
  descriptor.assignOrientations ( surf_kps.begin(), surf_kps.end() );
  descriptor.makeDescriptors ( surf_kps.begin(), surf_kps.end() );

  if ( surf_kps.size() == 0 )
  {
    return std::vector<cv::KeyPoint3D>();
  }

  int desc_len = surf_kps[0]._vec.size();
  int num_kp = surf_kps.size();

  std::vector<cv::KeyPoint> kps;
  cv::Mat1d descriptors( num_kp, desc_len );

  for ( int i=0; i<num_kp; i++ )
  {
    cv::KeyPoint kp;
    parallelsurf::KeyPoint &surf_kp = surf_kps[i];

    kp.angle = surf_kp._ori;
    kp.class_id = 0;
    kp.octave = 0;
    kp.pt = cv::Point2f( surf_kp._x, surf_kp._y );
    kp.response = surf_kp._score;
    kp.size = surf_kp._scale * 1.3595559868917 * 4.0;
    kps.push_back(kp);

    for ( int j=0; j<desc_len; j++ )
    {
      descriptors[i][j] = surf_kp._vec[j];
    }
  }

  return makeKp3d( kps, descriptors );
}

std::vector<cv::KeyPoint3D> getSiftKp( const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  Lowe::SIFT sift;
  sift.PeakThreshInit = 0.01 * t;

  Lowe::SIFT::KeyList lowe_kps;
  Lowe::SIFT::Image lowe_img( gray_img.rows, gray_img.cols );

  for ( int y=0; y<gray_img.rows; y++ )
  {
    for ( int x=0; x<gray_img.cols; x++ )
    {
      lowe_img.pixels[y][x] = gray_img.at<uchar>(y,x) / 255.0;
    }
  }

  lowe_kps = sift.getKeypoints( lowe_img );
  if ( lowe_kps.size() == 0 )
  {
    return std::vector<cv::KeyPoint3D>();
  }

  int desc_len = lowe_kps[0].ivec.size();
  int num_kp = lowe_kps.size();

  std::vector<cv::KeyPoint> kps;
  cv::Mat1d descriptors( num_kp, desc_len );

  for ( int i=0; i<num_kp; i++ )
  {
    cv::KeyPoint kp;
    Lowe::SIFT::Key& lowe_kp = lowe_kps[i];
    kp.angle = 0;
    kp.class_id = 0;
    kp.octave = 0;
    kp.pt = cv::Point2f( lowe_kp.col, lowe_kp.row );
    kp.response = lowe_kp.strength;
    kp.size = lowe_kp.scale * 1.3595559868917 * 4.0;
    kps.push_back(kp);
    for ( int j=0; j<desc_len; j++ )
    {
      descriptors[i][j] = lowe_kp.ivec[j];
    }
  }

  return makeKp3d( kps, descriptors );
}

std::vector<cv::KeyPoint3D> getDaftKp( daft_ns::DAFT::DetectorParams p, const cv::Mat& gray_img, const cv::Mat& depth_img, cv::Matx33f& K, float  t )
{
  std::vector<cv::KeyPoint3D> kp;
  p.det_threshold_ *= t;
  daft_ns::DAFT daft( p );
  daft.detect( gray_img, depth_img, K, kp );
  return kp;
}

void ExtractDetectorFile::extractKeypoints( GetKpFunc getKp, std::string name )
{
  uint32_t count = 1;
  bool first_image = true;

  cv::Mat first_kp_img;

  std::vector< ImageData >::iterator it;

  int it_step;
  std::vector< ImageData >::iterator it_end,it_begin;

  it_step = 1;
  it_begin = image_store_.begin();
  it_end = image_store_.end();

  float t=1;

  // !!! -1 because of the initial push_back in createTestFiles() ... !!!
  for (it = it_begin; it != it_end; it+=it_step)
  {
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
        std::vector<cv::KeyPoint3D> kp = getKp( gray_img, depth_img, K_, t );
        kp = filterKpMask(kp);
        kp_size = kp.size();

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
        if ( t < 0 ) t = 0;
        std::cout << " t_" << its+1 << " = " << t << std::endl;

        cv::Mat kp_img;
        cv::drawKeypoints3D(rgb_img, kp, kp_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //cv::imshow("KP", kp_img);
        //cv::waitKey(200);

        its++;
        std::cout << std::endl;
      }
    }

    std::stringstream s;
    s << "img" << count;
    count++;

    std::vector<cv::KeyPoint3D> kp = getKp( gray_img, depth_img, K_, t );
    std::cout << name << " " << s.str() << " #kp = " << kp.size() << std::endl;

    if(first_image)
    {
      kp = filterKpMask(kp);
      first_image = false;

      cv::drawKeypoints3D(rgb_img, kp, first_kp_img, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
      storeKeypoints(kp, s.str(), name, rgb_img );
    }
    else
    {
      cv::Mat first_kp_img_warped = rgb_img.clone();
      cv::warpPerspective( first_kp_img, first_kp_img_warped, cv::Mat(it->hom), cv::Size( rgb_img.cols, rgb_img.rows ) );

      printMat(it->hom);

      storeKeypoints(kp, s.str(), name, first_kp_img_warped );
    }

  }
}

void ExtractDetectorFile::printMat( cv::Matx33f M )
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


std::vector<cv::KeyPoint3D> ExtractDetectorFile::filterKpMask( std::vector<cv::KeyPoint3D> kp )
{
  // filter keypoints which dont fit the mask
  std::vector<cv::KeyPoint3D> kp_filtered;

  for(uint32_t i = 0; i < kp.size(); i++)
  {
    //std::cout << int(maskImage_( kp.at(i).pt.y, kp.at(i).pt.x )) << " ";
    // check for black spots in maskimage
    if( maskImage_.at<cv::Vec3b>(int(kp[i].pt.y),int(kp[i].pt.x))[0] < 128 )
    {
      kp_filtered.push_back(kp[i]);
    }
  }

  //std::cout << "Filtered Keypoints: " << kp_filtered.size() << "   Standard Keypoints: " << kp.size() << std::endl;
  //std::cout << std::endl;
  return kp_filtered;
}

void ExtractDetectorFile::extractAllKeypoints()
{
  daft_ns::DAFT::DetectorParams p;
  p.max_px_scale_ = 800;
  p.min_px_scale_ = 3;
  //p.base_scale_ = 0.02;
  //p.scale_levels_ = 1;
  p.det_threshold_ = 0.0109759;
  p.pf_threshold_ = 5;

  p.det_type_=p.DET_BOX;
  p.affine_=false;
  p.max_search_algo_ = p.MAX_FAST;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT-Fast" );

  p.det_type_=p.DET_BOX;
  p.affine_=true;
  p.max_search_algo_ = p.MAX_WINDOW;
  extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT Affine" );

  p.det_type_ = p.DET_9X9;
  p.max_search_algo_ = p.MAX_WINDOW;
  p.affine_ = false;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT" );

  p.det_type_ = p.DET_9X9;
  p.max_search_algo_ = p.MAX_WINDOW;
  p.affine_ = true;
  //extractKeypoints( boost::bind( &getDaftKp, p, _1,_2,_3,_4 ), "DAFT Affine" );

  //extractKeypoints( &getSurfKp, "SURF" );
  //extractKeypoints( &getSiftKp, "SIFT" );
}

void ExtractDetectorFile::storeKeypoints(std::vector<cv::KeyPoint3D> keypoints, std::string img_name, std::string extension, cv::Mat& rgb_img )
{
  if ( keypoints.size() == 0 )
  {
    throw;
  }
  std::vector< cv::KeyPoint3D >::iterator it;
  double_t ax, bx, ay, by, a_length, b_length, alpha_a, alpha_b;
  double_t A, B, C;

  std::string filePath = file_created_folder_ +  "/" + img_name + "." +extension;

  // open file
  std::fstream file;
  file.open(filePath.c_str(), std::ios::out);

  // header
  file << keypoints[0].desc.size()+1 << std::endl;
  file << keypoints.size() << std::endl;

  for ( it = keypoints.begin(); it != keypoints.end(); it++ )
  {
    //hack
    //it->affine_minor = it->affine_major;

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

    B = ( ( ax * ay * pow(b_length,2) + bx * by * pow(a_length,2)) ) / (pow(a_length,2) * pow(b_length,2) );

    C = ( pow(ay,2) * pow(b_length,2) + pow(by,2) * pow(a_length,2)) / (pow(a_length,2) * pow(b_length,2) );

    file << it->pt.x << "  " << it->pt.y << "  " << A << "  " << B << "  " << C;

    // write world scale as "feature component"
    // so keypoints of different size don't get matched
    if ( it->world_size != 0 )
    {
      float s_log = log2( it->world_size );
      file << " " << s_log*100;
    }
    else
    {
      file << " 0.0";
    }

    for ( unsigned i=0; i<it->desc.size(); i++ )
    {
      file << " " << it->desc[i];
    }

    file  << std::endl;

  }

  file.close();

  // draw keypoints
  cv::Mat kp_img;

  cv::drawKeypoints3D(rgb_img, keypoints, kp_img, cv::Scalar(255,255,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  cv::putText( kp_img, extension, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 5, CV_AA );
  cv::putText( kp_img, extension, cv::Point(10,40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 2, CV_AA );

  //cv::imshow("kp", kp_img);
  //cv::waitKey(100);

  std::stringstream s;
  s.width(3);
  s.fill('0');
  s << img_count;

  //std::string img_file_name = extra_folder_ + "/" + extension + " " + img_name + ".ppm";
  std::string img_file_name = extra_folder_ + "/" + s.str() + ".ppm";

  std::cout << "Writing " << img_file_name << std::endl;

  cv::imwrite( img_file_name, kp_img);
  img_count++;
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
    std::cout << "Wrong usage, Enter: " << argv[0] << " <folderName> .." << std::endl;
    return -1;
  }

  for ( int i=1; i<argc; i++ )
  {
    rgbd_evaluator::img_count = 0;
    std::string file_name(argv[i]);
    rgbd_evaluator::ExtractDetectorFile extract_detector_file(file_name);
  }

  std::cout << "Exiting.." << std::endl;
  return 0;
}

