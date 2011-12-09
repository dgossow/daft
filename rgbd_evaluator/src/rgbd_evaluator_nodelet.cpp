/*
* Copyright (C) 2011 David Gossow
*/


#include "rgbd_evaluator/rgbd_evaluator.h"

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// watch the capitalization carefully

namespace rgbd_evaluator
{
	class RgbdEvaluatorNodelet : public nodelet::Nodelet
  {
	private:
		rgbd_evaluator::RgbdEvaluator* rgbd_eval_;

	public:
		virtual void onInit()
		{
			ROS_INFO("Hello!");
			rgbd_eval_ = new rgbd_evaluator::RgbdEvaluator(getNodeHandle(), getPrivateNodeHandle());
		}

		~RgbdEvaluatorNodelet()
		{
			delete rgbd_eval_;
		}
  };
}

PLUGINLIB_DECLARE_CLASS(rgbd_evaluator, RgbdEvaluatorNodelet, rgbd_evaluator::RgbdEvaluatorNodelet, nodelet::Nodelet)

