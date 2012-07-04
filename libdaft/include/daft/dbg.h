/*
 * dbg.h
 *
 * Assorted debugging / visualization stuff
 *
 *  Created on: Jul 4, 2012
 *      Author: gossow
 */

#ifndef DBG_H_
#define DBG_H_

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/timer.hpp>

static boost::timer t;
static std::string dbg_msg;

#define DBG_OUT( SEQ ) std::cout << SEQ << std::endl
#define TIMER_STOP if(dbg_msg.length()!=0) { DBG_OUT( "++++++ " << dbg_msg << ": " << t.elapsed()*1000.0 << "ms ++++++" ); }
#define TIMER_START( MSG ) TIMER_STOP dbg_msg=MSG; t.restart();

namespace cv
{

/*
inline void imshow( std::string win_title, cv::Mat img )
{
  cv::Mat img2 = img;
  if ( ( img.type() == CV_32F ) || ( img.type() == CV_64F ) )
  {
    img.convertTo( img2, CV_8U, 255, 0.0);
  }

  //cv::imshow( "x"+ win_title, img2 );
  std::string im_f = "/tmp/img/"+win_title+".png";
  std::cout << "Writing " << im_f << std::endl;
  cv::imwrite( im_f, img2 );
}
*/

inline void imshow2( std::string win_title, cv::Mat img, int size = -1 )
{
  cv::Mat img2;
  if ( size > 0 )
  {
    cv::resize( img, img2, Size( size, size), 0, 0, INTER_NEAREST );
  }
  else
  {
    img2 = img;
  }
  imshow( win_title, img2 );
}

inline void imshowNorm( std::string win_title, cv::Mat1f img, int size = -1 )
{
  double minv,maxv;
  int tmp;
  cv::minMaxIdx( img, &minv, &maxv, &tmp, &tmp );
  std::cout << win_title << " min=" << minv << " max=" << maxv << std::endl;
  if ( minv == maxv )
  {
    imshow2( win_title, img/maxv, size );
  }
  else
  {
    imshow2( win_title, (img - minv) * (1.0 / (maxv-minv)), size );
  }
}

inline void imshowDxDy( std::string win_title, cv::Mat1f img, int size = 256 )
{
  cv::Mat1f dx( img.rows-1, img.cols-1 );
  cv::Mat1f dy( img.rows-1, img.cols-1 );
  for ( int i = 0; i < dx.rows; i++ )
  {
    for ( int j = 0; j < dx.cols; ++j )
    {
      dx(i,j) = img(i,j+1)-img(i,j);
      dy(i,j) = img(i+1,j)-img(i,j);
    }
  }
  imshowNorm(win_title+" dx",dx,size);
  imshowNorm(win_title+" dy",dy,size);
}


}

#endif /* DBG_H_ */
