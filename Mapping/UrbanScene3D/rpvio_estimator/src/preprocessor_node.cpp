#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "rpvio_estimator/PlaneSegmentation.h"

#include "OpticalFlowPropagator.h"

OpticalFlowPropagator propagator;


// thread sync stuff
std::condition_variable con_frame_buf;
std::condition_variable con_propagator;
std::mutex m_frame_buf;
std::mutex m_propagator;
std::mutex m_sent_frame_id;
std::queue<Frame> frame_buf;
int current_frame_id = -1;
int sent_frame_id = -2;

// ros stuff
ros::ServiceClient client;
ros::Publisher rgb_image_pub;
ros::Publisher pcd_pub;
ros::Publisher odom_pub;
ros::Publisher plane_mask_pub;

Frame current_frame;


void publish_processed(const ProcessedFrame &f) {
    // TODO: publish
    ROS_INFO("Publishing frame %d", f.frame_id);
    sensor_msgs::ImagePtr rgb_image_msg = cv_bridge::CvImage(f.pcd->header, sensor_msgs::image_encodings::BGR8,
                                                             f.rgb_img).toImageMsg();

    sensor_msgs::ImagePtr plane_mask_msg = cv_bridge::CvImage(f.pcd->header, sensor_msgs::image_encodings::BGR8,
                                                              f.plane_mask).toImageMsg();

    // publishing stuff
    ROS_INFO("RGB image ts: %f", rgb_image_msg->header.stamp.toSec());
    ROS_INFO("Plane mask ts: %f", plane_mask_msg->header.stamp.toSec());
    ROS_INFO("Point cloud ts: %f", f.pcd->header.stamp.toSec());
    ROS_INFO("Odometry ts: %f", f.odom->header.stamp.toSec());
    rgb_image_pub.publish(rgb_image_msg);
    plane_mask_pub.publish(plane_mask_msg);
    pcd_pub.publish(f.pcd);
    odom_pub.publish(f.odom);

}

cv::Mat run_plannercnn(const Frame &f) {
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(f.img_header, sensor_msgs::image_encodings::BGR8,
                                                       f.rgb_img).toImageMsg();

    sensor_msgs::ImagePtr building_mask_msg = cv_bridge::CvImage(f.img_header, sensor_msgs::image_encodings::BGR8,
                                                                 f.building_mask).toImageMsg();

    rpvio_estimator::PlaneSegmentation plane_seg_srv;
    plane_seg_srv.request.frame_id = f.frame_id;
    plane_seg_srv.request.rgb_image = *img_msg;
    plane_seg_srv.request.building_mask_image = *building_mask_msg;
    ROS_DEBUG("Sending frame %d", f.frame_id);
    cv::Mat plane_mask;
    if (client.call(plane_seg_srv)) {
        ROS_DEBUG("Got back frame %d", f.frame_id);
        plane_mask = cv_bridge::toCvCopy(plane_seg_srv.response.img, sensor_msgs::image_encodings::BGR8)->image;
    } else {
        ROS_ERROR("Can't get response from plannercnn service, frame %ld", plane_seg_srv.request.frame_id);
    }

    return plane_mask;
//    std::this_thread::sleep_for(std::chrono::milliseconds(250));
//    return f.rgb_img;
}


// thread: propagate `processed_frame` and publish
[[noreturn]] void propagate_and_publish() {
    ROS_INFO("propagate started %d", std::this_thread::get_id());
    Frame f;
    while (true) {

        std::unique_lock<std::mutex> lk(m_propagator);
        con_propagator.wait(lk, [&] {

            m_frame_buf.lock();
            m_sent_frame_id.lock();

            while (!frame_buf.empty() and frame_buf.front().frame_id < propagator.source_frame.frame_id) {
                frame_buf.pop();
            }

            bool ok = propagator.source_frame.frame_id != -1 and !frame_buf.empty();

            ok &= ((propagator.source_frame.frame_id == sent_frame_id) or (frame_buf.front().frame_id < sent_frame_id));
            if (ok) {
                f = frame_buf.front();
                frame_buf.pop();
            }

            m_sent_frame_id.unlock();
            m_frame_buf.unlock();
            return ok;
        });


        ProcessedFrame processed_f;
        if (f.frame_id == propagator.source_frame.frame_id) {
            processed_f = propagator.source_frame;
        } else {
            processed_f = ProcessedFrame(f);
            processed_f.plane_mask = propagator.propagate_farneback(f.rgb_img);
        }

        lk.unlock();
        publish_processed(processed_f);
    }

}


// thread: process frames
[[noreturn]] void process() {
    ROS_INFO("PLANERCNN started %d", std::this_thread::get_id());
    Frame f;
    while (true) {
        if (f.frame_id != -1) {
            ROS_INFO("Intitiating process for frame %d", f.frame_id);

            ProcessedFrame processed_f(f);
            processed_f.plane_mask = run_plannercnn(f);


            m_propagator.lock();
            propagator.reset(processed_f);
            ROS_INFO("Saved frame %d", processed_f.frame_id);
            m_propagator.unlock();
            con_propagator.notify_one();
        }
        std::unique_lock<std::mutex> lk(m_frame_buf);
        con_frame_buf.wait(lk, [] { return !frame_buf.empty(); });
        f = current_frame;
        lk.unlock();

        m_sent_frame_id.lock();
        sent_frame_id = f.frame_id;
        m_sent_frame_id.unlock();
        con_propagator.notify_one();

    }
}

void preprocessing_callback(
        const sensor_msgs::PointCloudConstPtr &features_msg,
        const nav_msgs::OdometryConstPtr &odometry_msg,
        const sensor_msgs::ImageConstPtr &img_msg,
        const sensor_msgs::ImageConstPtr &building_mask
) {
    Frame f;
    f.frame_id = ++current_frame_id;
    f.pcd = features_msg;
    f.odom = odometry_msg;
    f.rgb_img = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8)->image;
    f.building_mask = cv_bridge::toCvCopy(building_mask, sensor_msgs::image_encodings::BGR8)->image;
    f.img_header = img_msg->header;

    m_frame_buf.lock();
    frame_buf.push(f);
    m_frame_buf.unlock();
    current_frame = f;
    con_frame_buf.notify_one();

}


int main(int argc, char **argv) {
    ros::init(argc, argv, "rpvio_preprocessor");
    ros::NodeHandle n;

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    message_filters::Subscriber<sensor_msgs::PointCloud> sub_point_cloud(n, "/vins_estimator/point_cloud", 10);
    message_filters::Subscriber<nav_msgs::Odometry> sub_odometry(n, "/vins_estimator/odometry", 10);
//    ros::Subscriber sub = n.subscribe("/image", 10, preprocessing_callback);
    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, "/image", 10);
    message_filters::Subscriber<sensor_msgs::Image> sub_building_mask(n, "/mask", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;

    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_point_cloud, sub_odometry, sub_image,
                                                     sub_building_mask);
    sync.registerCallback(boost::bind(&preprocessing_callback, _1, _2, _3, _4));

    // declare service client
    client = n.serviceClient<rpvio_estimator::PlaneSegmentation>("plane_segmentation");

    // publishers
    rgb_image_pub = n.advertise<sensor_msgs::Image>("/image_processed", 10);
    plane_mask_pub = n.advertise<sensor_msgs::Image>("/plane_mask_processed", 10);
    pcd_pub = n.advertise<sensor_msgs::PointCloud>("/point_cloud_processed", 10);
    odom_pub = n.advertise<nav_msgs::Odometry>("/odometry_processed", 10);


    std::thread plannercnn_process(process), propagate_process(propagate_and_publish);
    ros::spin();
}
