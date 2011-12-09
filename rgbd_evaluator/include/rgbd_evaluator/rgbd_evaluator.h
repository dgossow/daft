/*
 * Copyright (C) 2011 David Gossow
 */

#ifndef rgbd_evaluator_h_
#define rgbd_evaluator_h_

#include "rgbd_evaluator/extract_rgbd_features.h"

#include <ros/ros.h>

#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

namespace rgbd_evaluator {

class RgbdEvaluator {
public:

	/** \brief Constructor */
	RgbdEvaluator(ros::NodeHandle comm_nh, ros::NodeHandle param_nh);
	virtual ~RgbdEvaluator();

private:

	void subscribe();
	void unsubscribe();

	void publishKpMarkers( const std::map< double, std::vector<rgbd_features::KeyPoint> > & keypoints_by_size,
			std::string frame_id, ros::Time stamp, std::string ns, int id, float r, float g, float b );

	// message callbacks
	void rgbdImageCb(const sensor_msgs::Image::ConstPtr rgb_img,
			const sensor_msgs::Image::ConstPtr depth_img, const sensor_msgs::CameraInfo::ConstPtr cam_info );

	ros::NodeHandle comm_nh_;

	message_filters::Subscriber<sensor_msgs::Image> rgb_img_sub_;
	message_filters::Subscriber<sensor_msgs::Image> depth_img_sub_;
	message_filters::Subscriber<sensor_msgs::CameraInfo> cam_info_sub_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo > RgbdSyncPolicy;

  message_filters::Synchronizer<RgbdSyncPolicy> rgbd_sync_;

	ros::Publisher pub_markers_;

	tf::TransformListener tf_listener_;

	std::set< double > used_scales_;

	ExtractRgbdFeatures extract_rgbd_features_;
	size_t last_keypoint_num_;

	std::map< std::string, float > object_sizes_;
};

}

#endif
