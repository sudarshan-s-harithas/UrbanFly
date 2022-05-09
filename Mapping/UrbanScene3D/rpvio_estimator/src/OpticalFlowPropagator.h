//
// Created by user on 2/9/22.
//

#ifndef RPVIO_ESTIMATOR_OPTICALFLOWPROPAGATOR_H
#define RPVIO_ESTIMATOR_OPTICALFLOWPROPAGATOR_H

#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Header.h>
#include <nav_msgs/Odometry.h>
#include <thread>

struct Frame {
    sensor_msgs::PointCloudConstPtr pcd = nullptr;
    nav_msgs::OdometryConstPtr odom = nullptr;
    std_msgs::Header img_header;
    cv::Mat rgb_img, building_mask;
    int frame_id = -1;

    Frame() {
        this->frame_id = -1;
    }
};

struct ProcessedFrame : Frame {
    cv::Mat plane_mask;

    ProcessedFrame(const Frame &f) {
        this->pcd = f.pcd;
        this->odom = f.odom;
        this->rgb_img = f.rgb_img;
        this->building_mask = f.building_mask;
        this->frame_id = f.frame_id;
    }

    ProcessedFrame() {
    }
};

class OpticalFlowPropagator {
public:
    OpticalFlowPropagator();

    ~OpticalFlowPropagator();

    ProcessedFrame source_frame;
    cv::Mat old_gray, old_plane_mask;
    cv::Mat grid;

    void reset(const ProcessedFrame &f);

    cv::Mat propagate_farneback(const cv::Mat &rgb_img);

};


#endif //RPVIO_ESTIMATOR_OPTICALFLOWPROPAGATOR_H
