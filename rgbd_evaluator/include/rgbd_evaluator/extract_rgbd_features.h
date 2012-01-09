/*
 * Copyright (C) 2011 David Gossow
 */

#ifndef extract_rgbd_features_h_
#define extract_rgbd_features_h_

#include "rgbd_evaluator/RgbdFeaturesConfig.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>

#include <rgbd_features/integral_image.h>
#include <rgbd_features/key_point.h>

#include <dynamic_reconfigure/server.h>

namespace rgbd_evaluator {

class ExtractRgbdFeatures {
public:

        /** \brief Constructor */
        ExtractRgbdFeatures(ros::NodeHandle comm_nh, ros::NodeHandle param_nh);
        virtual ~ExtractRgbdFeatures();

        // message callbacks
        void processPointCloud(const sensor_msgs::PointCloud2::ConstPtr& msg);
        void processCameraInfo(const sensor_msgs::CameraInfo::ConstPtr& msg);

        const std::vector<rgbd_features::KeyPoint> & keypoints() const { return keypoints_; }

private:

        void dynConfigCb(RgbdFeaturesConfig &config, uint32_t level);

        // core feature computation
        void computeFeatures(const sensor_msgs::PointCloud2::ConstPtr& msg);

        void imageSizeChanged();

        void fillIntegralImage();
        void fillIntensityImage();
        void fillDepthImage();
        void fillChromaImage();

        void calcScaleMap();

        void publishImage();
        void paintKeypoints(std::vector<rgbd_features::KeyPoint> &keypoints, int r = 0,
                        int g = 255, int b = 0);

        void paintInputImage();
        void paintDetectorImage();
        void paintInputAndDetectorImage();

        void setPixel(int x, int y, int r = 0, int g = 127, int b = 0);
        void rasterCircle(int x0, int y0, int radius, int r = 0, int g = 127,
                        int b = 0);

        void updateDetectorLimits();

        void getMinMax(double **image, double &min_val, double &max_val);

        dynamic_reconfigure::Server<RgbdFeaturesConfig> dyn_conf_srv_;
        dynamic_reconfigure::Server<RgbdFeaturesConfig>::CallbackType dyn_conf_cb_;

        sensor_msgs::PointCloud2::ConstPtr point_cloud_;

        rgbd_features::IntegralImage integral_image_;
        double **scale_map_;

        std::vector<rgbd_features::KeyPoint> keypoints_;

        double f_;
        int width_;
        int height_;

        double **input_image_;

        double **detector_image_;
        double detector_image_min_;
        double detector_image_max_;

        RgbdFeaturesConfig config_;

        image_transport::Publisher result_image_publisher_;
        sensor_msgs::Image result_image_;

        sensor_msgs::PointCloud2::ConstPtr last_point_cloud_;
};

}

#endif
