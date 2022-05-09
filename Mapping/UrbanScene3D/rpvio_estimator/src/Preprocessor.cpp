//
// Created by user on 2/9/22.
//

#include "Preprocessor.h"

#include <utility>


Preprocessor::Preprocessor(std::function<void(const OutputData &)> publisher, int gap) {
    this->publisher = std::move(publisher);
    this->runner_gap = gap;
    this->to_run_frame_id = 0;
}

void Preprocessor::add_data(const InputData &inp) {
    Data d;

    d.frame_id = ++current_frame_id;
    d.imu = inp.imu;
    d.img = cv_bridge::toCvCopy(inp.img, sensor_msgs::image_encodings::BGR8)->image;

    // TODO: implement queue pushing and running logic

    // just push in the queue
    m_data_buf.lock();

    data_buf.push(d);

    m_data_buf.unlock();
}




