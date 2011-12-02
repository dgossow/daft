/*
 * Copyright (C) 2011 David Gossow
 */

#ifndef rgbd_evaluator_h_
#define rgbd_evaluator_h_

#include "rgbd_evaluator/extract_rgbd_features.h"

#include <ros/ros.h>

#include <tf/transform_listener.h>

#include <sensor_msgs/PointCloud2.h>
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
	void pointCloudCb(const sensor_msgs::PointCloud2::ConstPtr& msg);
	void cameraInfoCb(const sensor_msgs::CameraInfo::ConstPtr& msg);

	ros::Subscriber sub_pointcloud2_;
	ros::Subscriber sub_camerainfo_;

	ros::Publisher pub_markers_;

	ros::NodeHandle comm_nh_;

	tf::TransformListener tf_listener_;

	std::set< double > used_scales_;

	ExtractRgbdFeatures extract_rgbd_features_;
	size_t last_keypoint_num_;

	std::map< std::string, float > object_sizes_;
};

}

#endif
