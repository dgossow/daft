/*
* Copyright (C) 2011 David Gossow
*/

#include "rgbd_evaluator/rgbd_evaluator.h"

#include <visualization_msgs/MarkerArray.h>

// ROS
#include <ros/ros.h>
#include <ros/package.h>

#include <cv.h>

namespace rgbd_evaluator
{

RgbdEvaluator::RgbdEvaluator ( ros::NodeHandle comm_nh, ros::NodeHandle param_nh ):
		comm_nh_( comm_nh ),
		tf_listener_ ( comm_nh, ros::Duration(30) ),
		extract_rgbd_features_( comm_nh, param_nh ),
		last_keypoint_num_(0)
{
  sub_camerainfo_ = comm_nh.subscribe ( "camera_info", 1, &RgbdEvaluator::cameraInfoCb, this );

  pub_markers_ = comm_nh.advertise<visualization_msgs::MarkerArray>( "rgbd_features", 0 );

  subscribe();
}


RgbdEvaluator::~RgbdEvaluator ()
{
}

void RgbdEvaluator::subscribe()
{
	  sub_pointcloud2_ = comm_nh_.subscribe ( "points", 1, &RgbdEvaluator::pointCloudCb, this );
	  ROS_INFO ("Subscribed to cloud on: %s", sub_pointcloud2_.getTopic ().c_str ());
}

void RgbdEvaluator::unsubscribe()
{
	sub_pointcloud2_ = ros::Subscriber();
	ROS_INFO ("Sleeping.");
}

void RgbdEvaluator::publishKpMarkers( const std::map< double, std::vector<rgbd_features::KeyPoint> > & keypoints_by_size,
		std::string frame_id, ros::Time stamp, std::string ns, int id, float r, float g, float b )
{
	std::map< double, std::vector<rgbd_features::KeyPoint> >::const_iterator it;

	visualization_msgs::MarkerArray marker_array;

	for ( it = keypoints_by_size.begin(); it != keypoints_by_size.end(); it++ )
	{
		double size = it->first * 2.0;
		const std::vector<rgbd_features::KeyPoint> & keypoints = it->second;

		visualization_msgs::Marker marker;
		marker.header.frame_id = frame_id;
		marker.header.stamp = stamp;
		marker.ns = ns;
		marker.id = id++;
		marker.type = visualization_msgs::Marker::SPHERE_LIST;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = size;
		marker.scale.y = size;
		marker.scale.z = size;
		marker.color.a = 0.5;
		marker.color.r = r;
		marker.color.g = g;
		marker.color.b = b;

		marker.points.reserve( keypoints.size() );

		for ( unsigned i = 0; i < keypoints.size(); i++ )
		{
			geometry_msgs::Point point;
			point.x = keypoints[i]._rx;
			point.y = keypoints[i]._ry;
			point.z = keypoints[i]._rz;
			marker.points.push_back( point );
		}

		ROS_INFO( "Scale %f: %i keypoints.", size, (int)(keypoints.size()) );

		marker_array.markers.push_back( marker );
		if ( keypoints.size() == 0 )
		{
			used_scales_.erase( size );
		}
	}

	pub_markers_.publish( marker_array );
}

void RgbdEvaluator::pointCloudCb ( const sensor_msgs::PointCloud2::ConstPtr& point_cloud )
{
	// detect keypoints

	extract_rgbd_features_.processPointCloud( point_cloud );

	// make cv::Mat out of RGB values

	unsigned char *rgb = ( unsigned char* ) & ( point_cloud->data ) [16];
	int point_step = point_cloud->point_step;
	int width = point_cloud->width;
	int height = point_cloud->height;

	cv::Mat image( height, width, CV_8UC3 );

	for ( int y=0; y<height; ++y )
	{
		for ( int x=0; x<width; ++x )
		{
			image.at<cv::Vec3b>(y,x)[0] = rgb[0];
			image.at<cv::Vec3b>(y,x)[1] = rgb[1];
			image.at<cv::Vec3b>(y,x)[2] = rgb[2];
			rgb += point_step;
		}
	}

	// get 3D points & publish Markers

	const std::vector<rgbd_features::KeyPoint> & keypoints = extract_rgbd_features_.keypoints();

	std::map< double, std::vector<rgbd_features::KeyPoint> > keypoints_by_size;

	for ( std::set< double >::iterator used_scales_it = used_scales_.begin(); used_scales_it != used_scales_.end(); used_scales_it++ )
	{
		keypoints_by_size[ *used_scales_it ] = std::vector<rgbd_features::KeyPoint>();
	}

	for ( unsigned i=0; i<keypoints.size(); i++ )
	{
		keypoints_by_size[ keypoints[i]._physical_scale ].push_back( keypoints[i] );
		used_scales_.insert( keypoints[i]._physical_scale );
	}

	publishKpMarkers( keypoints_by_size, point_cloud->header.frame_id, point_cloud->header.stamp, "current_kps", 0, 0, 1, 0 );
}

void RgbdEvaluator::cameraInfoCb ( const sensor_msgs::CameraInfo::ConstPtr& msg )
{
	extract_rgbd_features_.processCameraInfo( msg );
}

}

