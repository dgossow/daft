/*
 * fixedframepublisher.cpp
 *
 *  Created on: Nov 23, 2011
 *      Author: gossow
 */

#include "ar_fixed_frame/fixed_frame_publisher.h"

#include <boost/bind.hpp>

namespace ar_fixed_frame
{


FixedFramePublisher::FixedFramePublisher ( ros::NodeHandle comm_nh, ros::NodeHandle param_nh ):
				tf_listener_ ( comm_nh, ros::Duration(30) )
{
	tf_listener_.addTransformsChangedListener( boost::bind( &FixedFramePublisher::transformsChanged, this )  );
}

FixedFramePublisher::~FixedFramePublisher() {
	// TODO Auto-generated destructor stub
}

void FixedFramePublisher::transformsChanged()
{
	tf::StampedTransform marker_1_transform,marker_2_transform,marker_3_transform;

	try {
		tf_listener_.lookupTransform( "openni_camera", "marker_1", ros::Time(0), marker_1_transform );
		tf_listener_.lookupTransform( "openni_camera", "marker_2", ros::Time(0), marker_2_transform );
		tf_listener_.lookupTransform( "openni_camera", "marker_3", ros::Time(0), marker_3_transform );
	}
	catch ( std::runtime_error err )
	{
		ROS_ERROR_THROTTLE( 1.0, "Error while looking up transform: %s", err.what() );
		return;
	}

	btVector3 center = marker_1_transform.getOrigin() + marker_2_transform.getOrigin() + marker_3_transform.getOrigin();
	center /= 3.0;

	btVector3 u = marker_2_transform.getOrigin() - marker_1_transform.getOrigin();
	btVector3 v = marker_3_transform.getOrigin() - marker_1_transform.getOrigin();
	btVector3 w = u.cross(v);
	btVector3 v1 = w.cross( u );

	btMatrix3x3 basis;
	basis[0] = u.normalize();
	basis[1] = v1.normalize();
	basis[2] = w.normalize();
	basis=basis.transpose();

	tf::StampedTransform center_transform;
	center_transform.child_frame_id_ = "marker_center";
	center_transform.frame_id_ = "openni_camera";
	center_transform.setOrigin( center );
	center_transform.setBasis( basis );

	if ( marker_1_transform.stamp_ > marker_2_transform.stamp_ )
	{
		if ( marker_1_transform.stamp_ > marker_3_transform.stamp_ )
		{
			center_transform.stamp_ = marker_1_transform.stamp_;
		}
		else
		{
			center_transform.stamp_ = marker_3_transform.stamp_;
		}
	}
	else
	{
		if ( marker_2_transform.stamp_ > marker_3_transform.stamp_ )
		{
			center_transform.stamp_ = marker_2_transform.stamp_;
		}
		else
		{
			center_transform.stamp_ = marker_3_transform.stamp_;
		}
	}

	tf_broadcaster_.sendTransform( center_transform );
}

}

int main( int argc, char** argv )
{
  ros::init( argc, argv, "ar_fixed_frame_publisher" );

  ros::NodeHandle comm_nh(""); // for topics, services
  ros::NodeHandle param_nh("~");

  ar_fixed_frame::FixedFramePublisher *ffp = new ar_fixed_frame::FixedFramePublisher( comm_nh, param_nh );

  ros::spin();

  delete ffp;

  ROS_INFO( "Exiting.." );
  return 0;
}
