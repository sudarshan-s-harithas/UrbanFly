//
// Created by user on 2/9/22.
//

#include "OpticalFlowPropagator.h"

void OpticalFlowPropagator::reset(const ProcessedFrame &f) {
    this->source_frame = f;
    cv::cvtColor(f.rgb_img, this->old_gray, cv::COLOR_BGR2GRAY);
    this->old_plane_mask = f.plane_mask;
}

cv::Mat OpticalFlowPropagator::propagate_farneback(const cv::Mat &rgb_img) {
    cv::Mat gray_img;
    cv::cvtColor(rgb_img, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat flow(this->old_gray.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(this->old_gray, gray_img, flow, .5, 3, 15, 3, 5, 1.2, 0);

    cv::Mat map(flow.size(), CV_32FC2);

    cv::subtract(this->grid, flow, map);

    cv::Mat new_mask;
    cv::remap(this->old_plane_mask, new_mask, map, cv::Mat(), cv::INTER_LINEAR);

    // update old things
    this->old_gray = gray_img;
    this->old_plane_mask = new_mask;
    return new_mask;
//    std::this_thread::sleep_for(std::chrono::milliseconds(60));
//    return rgb_img;
}

OpticalFlowPropagator::OpticalFlowPropagator() {
    // NOTE: change below if dimension changes
    cv::Size s;
    s.height = 480;
    s.width = 640;
    this->grid = cv::Mat(s, CV_32FC2);
    for (int y = 0; y < grid.rows; y++) {
        for (int x = 0; x < grid.cols; x++) {
            grid.at<cv::Point2f>(y, x) = cv::Point2f(x, y);
        }
    }
}

OpticalFlowPropagator::~OpticalFlowPropagator() {

}

