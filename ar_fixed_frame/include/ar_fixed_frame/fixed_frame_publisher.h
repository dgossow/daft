/*
 * fixedframepublisher.h
 *
 *  Created on: Nov 23, 2011
 *      Author: gossow
 */

#ifndef FIXEDFRAMEPUBLISHER_H_
#define FIXEDFRAMEPUBLISHER_H_

#include <ros/ros.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

namespace ar_fixed_frame
{

class FixedFramePublisher
{
public:

	FixedFramePublisher( ros::NodeHandle comm_nh, ros::NodeHandle param_nh );
	virtual ~FixedFramePublisher();

	void transformsChanged();

private:

	tf::TransformListener tf_listener_;
	tf::TransformBroadcaster tf_broadcaster_;
};

}


#endif /* FIXEDFRAMEPUBLISHER_H_ */
