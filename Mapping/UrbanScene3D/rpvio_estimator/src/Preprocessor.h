//
// Created by user on 2/9/22.
//

#ifndef RPVIO_ESTIMATOR_PREPROCESSOR_H
#define RPVIO_ESTIMATOR_PREPROCESSOR_H

#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>


struct InputData {
    sensor_msgs::ImuPtr imu;
    sensor_msgs::ImagePtr img;
};

struct OutputData : InputData {
    sensor_msgs::ImagePtr plane_mask;
};

struct Data {
    sensor_msgs::ImuPtr imu;
    cv::Mat img;
    int frame_id;
};


class Preprocessor {
public:
    Preprocessor(std::function<void(const OutputData &)>, int gap = 5);

    void add_data(const InputData &);


    cv::Mat get_planercnn_mask(const cv::Mat &);

    ~Preprocessor();

private:
    int to_run_frame_id = 0;
    int current_frame_id = -1;
    int runner_gap = 5;
    std::function<void(const OutputData &)> publisher;

    std::mutex m_data_buf;
    std::queue<Data> data_buf;


};


#endif //RPVIO_ESTIMATOR_PREPROCESSOR_H
