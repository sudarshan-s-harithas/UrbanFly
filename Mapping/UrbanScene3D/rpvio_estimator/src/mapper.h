#pragma once

/**
 * To subscribe to
 *  mask cloud
 *  poses
 *  feature points (/current_cloud)
 * 
 * Maintain header wise 
 *  images or mask_clouds
 *  poses
 * 
 * Maintain window wise
 *  sparse feature points (/current_cloud)
 * 
 * Algorithm:
 *  on mask_cloud
 *      hash w.r.t header
 *  on pose
 *      hash w.r.t header
 *  on feature cloud
 *      optimize
 *      reconstruct
 *      maintain plane id tracks
 **/

// first things first
/**
 * simple log from mapper node
 * subscribe to point_cloud and color them based on plane id and publish coloured point cloud
 * subscribe to header-wise pose
 * subscribe to header-wise mask_cloud
 * (
 * Write a common callback using time synchronizer
 * in the callback, 
 *  add all the messages to a buffer
 *  maintain current and previous plane ids
 *  maintain plane id vs plane params (first measurement)
 * 
 * Also publish all the plane clouds (), using mask clouds in the buffer
 * When a plane goes out of view,
 *  Optimize all the planes, update the map
 *  
 **/
#include "parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <chrono>

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <set>
#include <algorithm>
#include <condition_variable>
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "utility/color_ids.h"
#include "vp_utils.h"

using namespace std;
using namespace Eigen;

ros::Publisher pub_plane_cloud;
ros::Publisher marker_pub;
ros::Publisher ellipse_pub;
ros::Publisher frame_pub;
ros::Publisher cent_pub;
ros::Publisher ma_pub;
ros::Publisher frame_pub2;
ros::Publisher masked_im_pub;
ros::Publisher lgoal_pub;
ros::Publisher dense_pub;

struct Plane
{
    Vector4d plane;
    int plane_id;
    bool is_initialized = false;
    set<int> feature_ids;
    double best_fit_error = 100000.0;
    int best_num_of_inliers = 4;
    bool should_update = false;
};

struct PlaneFeature
{
    Vector3d point;
    int measurement_count = 0;
    int plane_id;
    bool is_outlier = false;
};

map<double, vector<Vector4d>> plane_measurements;
map<int, Plane> mPlaneFeatureIds;
map<int, PlaneFeature> mFeatures;
int plane_id_counter = 1000;

// for ros message sync
struct SubMessages
{
    sensor_msgs::PointCloudConstPtr features_msg;
    nav_msgs::OdometryConstPtr odometry_msg;
    // sensor_msgs::ImageConstPtr img_msg;
    sensor_msgs::ImageConstPtr mask_msg;
};
queue<SubMessages> sm_queue;
condition_variable sm_cond;
mutex sm_mutex;

/**
 * Implements vector1.dot(vector2) == 0 constraint
 * residual size: 1
 **/
struct OrthogonalConstraint {
    template <typename T>
    bool operator()(
        const T* const vector_array1,
        const T* const vector_array2,
        T* residual
    ) const {
        Eigen::Matrix<T, 3, 1> v1;
        v1 << vector_array1[0], vector_array1[1], vector_array1[2];
        
        Eigen::Matrix<T, 3, 1> v2;
        v2 << vector_array2[0], vector_array2[1], vector_array2[2];

        residual[0] = T(100) * ceres::DotProduct(v1.normalized().data(), v2.normalized().data());
        return true;
    }
};


/**
 * Implements vector1.dot(vector2) == 0 constraint
 * residual size: 1
 **/
struct OrthogonalConstraintQuat {
    template <typename T>
    bool operator()(
        const T* const vector_array1,
        const T* const vector_array2,
        T* residual
    ) const {
        Eigen::Quaternion<T> v1(vector_array1[3], vector_array1[0], vector_array1[1], vector_array1[2]);
        v1.normalize();

        Eigen::Quaternion<T> v2(vector_array2[3], vector_array2[0], vector_array2[1], vector_array2[2]);
        v2.normalize();

        residual[0] = T(100) * v1.vec().normalized().dot(v2.vec().normalized());
        return true;
    }
};

/**
 * Implements vector1.cross(vector2) == zero_vec(tor constraint
 * residual size: 3
 **/
struct ParallelConstraint {
    template <typename T>
    bool operator()(
        const T* const vector_array1,
        const T* const vector_array2,
        T* residual
    ) const {
        T cross[3];
        Eigen::Matrix<T, 3, 1> v1;
        v1 << vector_array1[0], vector_array1[1], vector_array1[2];
        
        Eigen::Matrix<T, 3, 1> v2;
        v2 << vector_array2[0], vector_array2[1], vector_array2[2];

        ceres::CrossProduct(v1.normalized().data(), v2.normalized().data(), cross);

        residual[0] = T(100) * cross[0];
        residual[1] = T(100) * cross[1];
        residual[2] = T(100) * cross[2];
        return true;
    }
};

/**
 * Implements horizontal normal constraint a.k.a vertical plane
 * residual size: 1
 **/
struct VerticalPlaneConstraint {
    template <typename T>
    bool operator()(
        const T* const vector_array1,
        T* residual
    ) const {        
        Eigen::Quaternion<T> v1(vector_array1);
        residual[0] = T(100) * v1.normalized().vec()(2);
        return true;
    }
};

// Plane measurement Constraint
// residual size: 4
struct PlaneMeasurementConstraint {
    PlaneMeasurementConstraint(Quaterniond &plane_meas): plane_meas_(plane_meas) {}
    
    template <typename T>
    bool operator()( 
        const T* const plane, T* residual
    ) const {
        Eigen::Quaternion<T> plane_quat(plane[3], plane[0], plane[1], plane[2]);
        plane_quat.normalize();
        Eigen::Quaternion<T> meas_quat = plane_meas_.cast<T>();;
        meas_quat.normalize();
        
        // residual[0] = meas_quat.x() - plane_quat.x();
        // residual[1] = meas_quat.y() - plane_quat.y();
        // residual[2] = meas_quat.z() - plane_quat.z();
        // residual[3] = meas_quat.w() - plane_quat.w();

        residual[0] = T(100) * (meas_quat.coeffs().normalized() - plane_quat.coeffs().normalized()).norm();
        return true;
    }
    
    Quaterniond plane_meas_;
};

/**
 * Implements local parameterization of unit normal
 **/
struct UnitNormalParameterization {
    template <typename T>
    bool operator()(
        const T* x, const T* delta, T* x_plus_delta
    ) const {
        Eigen::Map<const Eigen::Matrix<T, 2, 1>> x_(x);
        Eigen::Map<const Eigen::Matrix<T, 2, 1>> delta_(delta);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> x_plus_delta_(x_plus_delta);

        x_plus_delta_ = x_ + delta_;
        x_plus_delta_.normalize();

        return true;
    }
};

void publish_plane_cloud(
    vector<sensor_msgs::PointCloudConstPtr> &mask_clouds,
    vector<nav_msgs::OdometryConstPtr> &odometry_msgs,
    map<int, Vector4d> plane_params
){
    sensor_msgs::PointCloud plane_cloud;
    sensor_msgs::ChannelFloat32 colors;
    colors.name = "rgb";

    map<int, Isometry3d> plane2world_isos;
    map<int, Vector4d> plane_seg_params; // min_z, max_z, min_y, max_y
        
    vector<int> cur_pids;
    for (map<int, Vector4d>::iterator it = plane_params.begin(); it != plane_params.end(); ++it){
        Vector4d pp = it->second;

        Vector3d normal;
        normal << pp(0), pp(1), pp(2);
        double d = -pp(3)/normal.norm();
        normal.normalize();
        
        if (fabs(d) < 100) {
            cur_pids.push_back(it->first);
            
            // Find nearest point on the plane (as center)
            Vector3d center = d * normal;

            Vector3d x_axis;
            x_axis << 1.0, 0.0, 0.0;

            // Compute the rotation of x-axis w.r.t normal
            double cos_theta = x_axis.dot(normal);
            double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

            Isometry3d plane2world;
            plane2world.translation() = center;
            plane2world.linear() <<
                cos_theta, -sin_theta, 0.0,
                sin_theta, cos_theta, 0.0,
                0.0, 0.0, 1.0; 
            
            plane2world_isos[it->first] = plane2world;
            plane_seg_params[it->first] = Vector4d::Zero();
        }
        // ROS_INFO("---------------CURRENT PIDs----------------ID = %d --------------------", it->first);
    }

    /**
     * Plan:
     * 
     * Compute isometry for each plane frist
     * 
     * For each point:
     *  Transform the world point to its plane coord sys
     *  Check if its on border
     *  Update plane segment params
     *  Compute border points from plane segment params
     *  Transform border points to world sys
     *  Create line strip from border points and publish
     **/

    for(int qi = mask_clouds.size()-1; qi < mask_clouds.size(); qi++) {
        // publish plane cloud
        sensor_msgs::PointCloudConstPtr mask_cloud = mask_clouds[qi];
        nav_msgs::OdometryConstPtr odometry_msg = odometry_msgs[qi];
        plane_cloud.header = mask_cloud->header;

        ROS_DEBUG("-----------Computing FIXED transforms--------------");
        // Retrieve pose from odometry message
        Isometry3d Tic;
        Tic.linear() = RIC[0];
        Tic.translation() = TIC[0];
        ROS_DEBUG("-----------Computed FIXED transforms DONE--------------");

        ROS_DEBUG("-----------Computing odometry transforms--------------");

        Vector3d trans;
        trans <<
            odometry_msg->pose.pose.position.x,
            odometry_msg->pose.pose.position.y,
            odometry_msg->pose.pose.position.z;

        double quat_x = odometry_msg->pose.pose.orientation.x;
        double quat_y = odometry_msg->pose.pose.orientation.y;
        double quat_z = odometry_msg->pose.pose.orientation.z;
        double quat_w = odometry_msg->pose.pose.orientation.w;
        Quaterniond quat(quat_w, quat_x, quat_y, quat_z);

        Isometry3d Ti;
        Ti.linear() = quat.normalized().toRotationMatrix();
        Ti.translation() = trans;
        
        ROS_DEBUG("-----------Computed odometry transforms DONE--------------");

        map<int, int> pid2HEX;
        pid2HEX[39] = 0x0000FF;// cv::Scalar(255,0,0);
        pid2HEX[66] = 0xFF00FF;// cv::Scalar(255,0,255);
        pid2HEX[91] = 0x00FF00;// cv::Scalar(0,255,0);
        pid2HEX[130] = 0xFF0000;// cv::Scalar(0,0,255);
        pid2HEX[162] = 0x00FFFF;// cv::Scalar(255,255,0);
        pid2HEX[175] = 0xFFFF00;// cv::Scalar(0,255,255);

        // Back-project all mask points using odometry
        ROS_DEBUG("=============Back projecting mask points to planes DONE=============");
        for (unsigned int i = 0; i < mask_cloud->points.size(); i++) {
            int ppid = mask_cloud->channels[1].values[i];
            if (find(cur_pids.begin(), cur_pids.end(), ppid) != cur_pids.end()) {
                Vector4d pp = plane_params[ppid];

                Vector3d normal;
                normal << pp(0), pp(1), pp(2);
                double d = -pp(3)/normal.norm();
                normal.normalize();

                // if (fabs(d) < 100) {
                    double lambda = 0.0;
                    double lambda2 = 0.0;

                    geometry_msgs::Point32 m;
                    m = mask_cloud->points[i];
                    Vector3d c_point(m.x, m.y, m.z);

                    Vector4d pp_ci = (Ti * Tic).matrix().transpose() * pp.normalized();

                    Vector3d normal_ci;
                    normal_ci << pp_ci(0), pp_ci(1), pp_ci(2);
                    double d_ci = -pp_ci(3)/normal_ci.norm();
                    normal_ci.normalize();
                    
                    // Vector3d ray;
                    // ray << c_point(0), c_point(1), c_point(2);
                    // ray.normalize();

                    // if (fabs(normal_ci.dot(ray)) < 0.6)
                    //     continue;

                    Vector4d pp2 = pp.normalized();
                    pp2.head<3>() *= -1;
                    Vector4d pp_ci2 = (Ti * Tic).matrix().transpose() * pp2.normalized();

                    Vector3d normal_ci2;
                    normal_ci2 << pp_ci2(0), pp_ci2(1), pp_ci2(2);
                    double d_ci2 = -pp_ci2(3)/normal_ci2.norm();
                    normal_ci2.normalize();

                    lambda2 = -fabs(d_ci2) / (normal_ci2.dot(c_point));
                    lambda = -fabs(d_ci) / (normal_ci.dot(c_point));

                    if ((lambda2 * c_point)(2) < 0) {
                        c_point = lambda * c_point;
                                            
                    } else {
                        c_point = lambda2 * c_point;
                    }

                    // Vector3d ray;
                    // ray << c_point(0), c_point(1), c_point(2);
                    // ray.normalize();
                    Vector3d principal_ray;
                    principal_ray << 0.0, 0.0, 1.0;

                    if ((fabs(normal_ci.dot(principal_ray)) < 0.2) || (fabs(normal_ci2.dot(principal_ray)) < 0.2))
                        continue;
                    
                    // Transform c_point (current camera) to imu (current IMU)
                    Vector3d i_point = Tic.rotation() * c_point + Tic.translation();

                    // Transform current imu point to world imu
                    Vector3d i0_point = (Ti.rotation() * i_point) + Ti.translation();

                    Vector3d w_pts_i = i0_point;

                    geometry_msgs::Point32 p;
                    p.x = w_pts_i(0);
                    p.y = w_pts_i(1);
                    p.z = w_pts_i(2);
                    plane_cloud.points.push_back(p);
                    
                    int rgb = mask_cloud->channels[0].values[i];
                    // int rgb = pid2HEX[ppid];
                    float float_rgb = *reinterpret_cast<float*>(&rgb);
                    colors.values.push_back(float_rgb);

                    // Check if it's on border
                    Vector3d p_pt = plane2world_isos[ppid].inverse() * w_pts_i;
                    Vector4d borders = plane_seg_params[ppid];

                    if (p_pt(1) < borders(0)) // min y
                        plane_seg_params[ppid](0) = p_pt(1);
                    else if (p_pt(1) > borders(1)) // max y
                        plane_seg_params[ppid](1) = p_pt(1);
                    else if (p_pt(2) < borders(2)) // min z
                        plane_seg_params[ppid](2) = p_pt(2);
                    else if (p_pt(2) > borders(3)) // max z
                        plane_seg_params[ppid](3) = p_pt(2);
                // }
            }
        }
    }
    // publish current point cloud
    plane_cloud.channels.push_back(colors);
    pub_plane_cloud.publish(plane_cloud);

    map<int, vector<Vector3d>> plane_border_points;
    visualization_msgs::Marker line_list;

    line_list.header = plane_cloud.header;
    // line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;

    line_list.id = 2;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    // LINE_LIST markers use only the x component of scale, for the line width
    line_list.scale.x = 0.5;

    // Line list is red
    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    sensor_msgs::PointCloud frame_cloud;
    sensor_msgs::ChannelFloat32 p_ids;

    // Compute border points for each plane
    for (map<int, Vector4d>::iterator it = plane_seg_params.begin(); it != plane_seg_params.end(); ++it){
        vector<geometry_msgs::Point> b_pts;
        
        double y_min = it->second(0);
        double y_max = it->second(1);
        double z_min = it->second(2);
        double z_max = it->second(3);

        // 0: (y_min, z_min)
        Vector3d bottom_left;
        bottom_left << 0, y_min, z_min;
        Vector3d bottom_left_w = plane2world_isos[it->first] * bottom_left;
        geometry_msgs::Point bl_pt;
        bl_pt.x = bottom_left_w(0);
        bl_pt.y = bottom_left_w(1);
        bl_pt.z = bottom_left_w(2);
        b_pts.push_back(bl_pt);

        // 1: (y_min, z_max)
        Vector3d top_left;
        top_left << 0, y_min, z_max;
        Vector3d top_left_w = plane2world_isos[it->first] * top_left;
        geometry_msgs::Point tl_pt;
        tl_pt.x = top_left_w(0);
        tl_pt.y = top_left_w(1);
        tl_pt.z = top_left_w(2);
        b_pts.push_back(tl_pt);

        // 2: (y_max, z_max)
        Vector3d top_right;
        top_right << 0, y_max, z_max;
        Vector3d top_right_w = plane2world_isos[it->first] * top_right;
        geometry_msgs::Point tr_pt;
        tr_pt.x = top_right_w(0);
        tr_pt.y = top_right_w(1);
        tr_pt.z = top_right_w(2);
        b_pts.push_back(tr_pt);

        // 3: (y_max, z_min)
        Vector3d bottom_right;
        bottom_right << 0, y_max, z_min;
        Vector3d bottom_right_w = plane2world_isos[it->first] * bottom_right;
        geometry_msgs::Point br_pt;
        br_pt.x = bottom_right_w(0);
        br_pt.y = bottom_right_w(1);
        br_pt.z = bottom_right_w(2);
        b_pts.push_back(br_pt);

        // 0 -> 1
        line_list.points.push_back(b_pts[0]);
        line_list.points.push_back(b_pts[1]);

        // 1 -> 2
        line_list.points.push_back(b_pts[1]);
        line_list.points.push_back(b_pts[2]);

        // 2 -> 3
        line_list.points.push_back(b_pts[2]);
        line_list.points.push_back(b_pts[3]);

        // 3 -> 0
        line_list.points.push_back(b_pts[3]);
        line_list.points.push_back(b_pts[0]);

        // 0: (y_min, z_min)
        geometry_msgs::Point32 bl_pt32;
        bl_pt32.x = bl_pt.x;
        bl_pt32.y = bl_pt.y;
        bl_pt32.z = bl_pt.z;
        frame_cloud.points.push_back(bl_pt32);
        p_ids.values.push_back(it->first);

        // 1: (y_min, z_max)
        geometry_msgs::Point32 tl_pt32;
        tl_pt32.x = tl_pt.x;
        tl_pt32.y = tl_pt.y;
        tl_pt32.z = tl_pt.z;
        frame_cloud.points.push_back(tl_pt32);
        p_ids.values.push_back(it->first);

        // 2: (y_max, z_max)
        geometry_msgs::Point32 tr_pt32;
        tr_pt32.x = tr_pt.x;
        tr_pt32.y = tr_pt.y;
        tr_pt32.z = tr_pt.z;
        frame_cloud.points.push_back(tr_pt32);
        p_ids.values.push_back(it->first);

        // 3: (y_max, z_min)
        geometry_msgs::Point32 br_pt32;
        br_pt32.x = br_pt.x;
        br_pt32.y = br_pt.y;
        br_pt32.z = br_pt.z;
        frame_cloud.points.push_back(br_pt32);
        p_ids.values.push_back(it->first);
    }
    frame_cloud.channels.push_back(p_ids);

    marker_pub.publish(line_list);
    frame_pub.publish(frame_cloud);
}

void processMask(cv::Mat &input_mask, cv::Mat &output_mask)
{
    // Convert RGB mask to CV_32U (32-bit single channel image)
    for (int i = 0; i < input_mask.rows; i++) {
        for (int j = 0; j < input_mask.cols; j++) {
            cv::Vec3b colors = input_mask.at<cv::Vec3b>(i, j);
            
            output_mask.at<uchar>(i, j) = (uchar)color2id(colors[0], colors[1], colors[2]);
        }
    }
}

/**
 * @brief Fills holes in the masks
 * 
 * @param input_mask 
 * @param output_mask 
 */
cv::Mat fillMaskHoles(cv::Mat input_mask, cv::Scalar color)
{
    cv::Mat binary_segment;
    cv::inRange(input_mask, color, color, binary_segment);

    input_mask.setTo(cv::Scalar(0, 0, 0), binary_segment);

    cv::Mat result_mask;
    cv::bitwise_xor(input_mask, color, result_mask);

    cv::morphologyEx(binary_segment, binary_segment, cv::MORPH_CLOSE, cv::Mat());
    cv::morphologyEx(binary_segment, binary_segment, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::erode(binary_segment, binary_segment, cv::Mat::ones(3, 3, CV_8U), cv::Point(-1, -1), 2);

    cv::Mat color_segment(input_mask.rows, input_mask.cols, CV_8UC3, color);

    cv::Mat cs;
    cv::cvtColor(binary_segment, cs, cv::COLOR_GRAY2RGB);

    cv::Mat inv_cs;
    cv::bitwise_xor(cs, cv::Scalar(255, 255, 255), inv_cs);

    cv::Mat re_mask;
    cv::bitwise_and(input_mask, inv_cs, re_mask);

    cv::Mat ccs;
    cv::bitwise_and(color_segment, cs, ccs);

    cv::bitwise_or(ccs, re_mask, result_mask);

    return result_mask;
}

/**
 * @brief Loop through each pixel to process mask of it's color
 * 
 * @param input_mask 
 * @param output_mask 
 */
cv::Mat processMaskSegments(cv::Mat input_mask)
{
    std::vector<unsigned int> processed_colors;

    auto tpcl_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < input_mask.rows; i++) {
        for (int j = 0; j < input_mask.cols; j++) {
            cv::Vec3b colors = input_mask.at<cv::Vec3b>(i, j);

            // Find hex representation of the color
            unsigned int hex_color = ((colors[0] & 0xff) << 16) + ((colors[1] & 0xff) << 8) + (colors[2] & 0xff);

            if(find(processed_colors.begin(), processed_colors.end(), hex_color) != processed_colors.end())
                continue;

            cv::Scalar color(colors[0], colors[1], colors[2]);
            processed_colors.push_back(hex_color);

            input_mask = fillMaskHoles(input_mask, color);
        }
    }
            
    auto tpcl_end = std::chrono::high_resolution_clock::now();
    double elapsed_timepcl_ms = std::chrono::duration<double, std::milli>(tpcl_end-tpcl_start).count();
    ROS_INFO("time taken for mask processing  is %g ms", elapsed_timepcl_ms);

    return input_mask;
}

double get_absolute_point_plane_distance(Vector3d point, Vector4d plane)
{
    Vector4d plane1(plane[0], plane[1], plane[2], plane[3]);
    Vector4d plane2(-plane[0], -plane[1], -plane[2], plane[3]);
    Vector4d plane3 = -plane1;
    Vector4d plane4 = -plane2;
    
    return min(
        min(
            fabs(plane1.dot(point.homogeneous())),
            fabs(plane2.dot(point.homogeneous()))
        ),
        min(            
            fabs(plane3.dot(point.homogeneous())),
            fabs(plane4.dot(point.homogeneous()))
        )
    );
    // return plane.dot(point.homogeneous());
}

int get_plane_id(int u, int v, cv::Mat &mask)
{
    int plane_id = 0;

    if ((u > 0) && (v > 0) && (u < mask.cols) && (v < mask.rows)) {
        cv::Vec3b colors = mask.at<cv::Vec3b>(v, u);
        
        plane_id = color2id(colors[0], colors[1], colors[2]);
    }

    return plane_id;
}

cv::Scalar hex2CvScalar(unsigned long hex)
{
    int r = ((hex >> 16) & 0xFF);
    int g = ((hex >> 8) & 0xFF);
    int b = ((hex) & 0xFF);

    cv::Scalar hex_color(r, g, b);

    return hex_color;
}

/**
 * Draws the plane segments on an image, 
 * coloured based on normal directions (that are calculated using vanishing points)
 **/
map<int, int> drawVPQuads( cv::Mat &img, std::vector<KeyLine> &lines, std::vector<std::vector<int> > &clusters, cv::Mat &seg_mask)
{
	map<int, Vector3d> plane_vplines;
    map<int, int> plane_normals;

    cv::Mat mask(ROW, COL, CV_8UC1, cv::Scalar(0));
    img.setTo(cv::Scalar(0, 0, 0));
    processMask(seg_mask, mask);

	std::vector<cv::Scalar> plane_colors( 2 );
	plane_colors[0] = cv::Scalar( 0, 0, 255 );
	plane_colors[1] = cv::Scalar( 255, 0, 0 );
	// plane_colors[2] = cv::Scalar( 0, 255, 0 );
	
	for ( size_t i = 0; i < clusters.size(); ++i )
	{
		for ( size_t j = 0; j < clusters[i].size(); ++j )
		{
			size_t idx = clusters[i][j];

			cv::Point pt_s = cv::Point( lines[idx].startPointX, lines[idx].startPointY );
			cv::Point pt_e = cv::Point( lines[idx].endPointX, lines[idx].endPointY );
			cv::Point pt_m = ( pt_s + pt_e ) * 0.5;
			
			int plane_id = get_plane_id((int)pt_m.x, (int)pt_m.y, seg_mask);
			if (plane_vplines.find(plane_id) == plane_vplines.end()){
				plane_vplines[plane_id] = Vector3d::Zero();
			}
			plane_vplines[plane_id](i)++;
		}
	}

	for (auto pvlines: plane_vplines)
	{
		if (
            (pvlines.first == 0)
            || (pvlines.first == 39)
        )
			continue;
		
		cv::Mat mask_img = mask.clone();
		cv::Mat mask = mask_img == pvlines.first;
		cv::Mat mask_filled(ROW, COL, CV_8UC3, cv::Scalar(0,0,0));
                
		int colour_id = pvlines.second[0] > pvlines.second[2] ?  0 : 1;
		mask_filled.setTo(plane_colors[colour_id], mask);
		
		cv::addWeighted( img, 1.0, mask_filled, 1.0, 0.0, img);
		// cv::imwrite("masked_image"+to_string(im_id)+".png", img);
        plane_normals[pvlines.first] = colour_id;
	}

    // cv::addWeighted( img, 0.0, seg_mask, 1.0, 0.0, img);
    // cv::imwrite("masked_image"+to_string(im_id)+".png", img);

    im_id++;

    return plane_normals;	
}

void draw_quad(cv::Mat &image, cv::Mat mask_image, int plane_id)
{
    cv::Mat plane_mask;

    unsigned long hex = id2color(plane_id);
    int r = ((hex >> 16) & 0xFF);
    int g = ((hex >> 8) & 0xFF);
    int b = ((hex) & 0xFF);

    cv::Scalar id_color(r, g, b);
    cv::inRange(mask_image, id_color, id_color, plane_mask);

    cv::Mat dilated;
    cv::dilate(plane_mask, dilated, cv::Mat(), cv::Point(-1, -1), 5);
    cv::erode(dilated, dilated, cv::Mat(), cv::Point(-1, -1), 7);
    // ROS_INFO("Dilated the plane mask with id %d", plane_id);

    // find contours
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours( dilated, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
    // std::cout << "largest contour has " << contours[0].size() << "points" << std::endl;
    // ROS_INFO("Found %d contours", (int)contours.size());

    if (contours.size() > 0)
    {
        // ROS_INFO("Largest contour has %d points", (int)contours[0].size());

        // simplify contours
        double epsilon = 0.01*cv::arcLength(contours[0], true);
        vector<cv::Point> approx_contour;
        cv::approxPolyDP(contours[0], approx_contour, epsilon, true);
        // std::cout << "simplified contour has" << approx_contour.size() << "points" << std::endl;
        // ROS_INFO("Simplified contour has %d points", (int)approx_contour.size());
        
        if (approx_contour.size() >= 4) {
            // Draw the simplified contour
            vector<vector<cv::Point> > approx_contours;
            approx_contours.push_back(approx_contour);
            // ROS_INFO("Drawing the contour");
            cv::drawContours(image, approx_contours, 0, id_color, cv::FILLED);
        }
    }
}

void draw_quads(cv::Mat &image, cv::Mat mask_image, vector<int> plane_ids)
{
    for (int i = 0; i < plane_ids.size(); i++)
    {
        draw_quad(image, mask_image, plane_ids[i]);
    }
}

Vector3d project_point_to_plane(Vector3d point, Vector4d plane)
{
    Vector3d normal = plane.head<3>();
    double d = plane[3];

    double lambda = (-d - point.dot(normal)) / normal.norm();

    return point + (lambda * normal);
}   

void compute_vertices_from_planes(Vector3d bound_point, vector<Vector4d> bound_planes, vector<Vector3d> &vertices)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    // Assuming planes are in this order: left, right, front back

    // There are four bound planes assuming they are vertical
    Vector4d plane1 = bound_planes[0];
    Vector4d plane2 = bound_planes[1];
    Vector4d plane3 = bound_planes[2];
    Vector4d plane4 = bound_planes[3];

    // Project to left most plane
    Vector3d left_point = project_point_to_plane(bound_point, plane1);
    Vector3d left_front = project_point_to_plane(left_point, plane3);
    Vector3d left_back = project_point_to_plane(left_point, plane4);

    // Project to right most plane
    Vector3d right_point = project_point_to_plane(bound_point, plane2);
    Vector3d right_front = project_point_to_plane(right_point, plane3);
    Vector3d right_back = project_point_to_plane(right_point, plane4);

    vertices.push_back(left_front);
    vertices.push_back(left_back);
    vertices.push_back(right_front);
    vertices.push_back(right_back);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    ROS_INFO("time taken for computing plane vertices is %g ms", elapsed_time_ms);
}

bool fit_cuboid_to_point_cloud(Vector4d plane_params, vector<Vector3d> points, vector<geometry_msgs::Point> &vertices, vector<Vector3d> &normal_vectors)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    Vector3d normal = plane_params.head<3>(); 
    Vector3d vertical(0, 1, 0);
    Vector3d horizontal = normal.cross(vertical).normalized();

    Vector3d min_n_pt;
    Vector3d min_h_pt;
    Vector3d min_v_pt;
    double min_n_d = 10000;
    double min_h_d = 10000;
    double min_v_d = 10000;

    Vector3d max_n_pt;
    Vector3d max_h_pt;
    Vector3d max_v_pt;
    double max_n_d = -10000;
    double max_h_d = -10000;
    double max_v_d = -10000;

    for (int i = 0; i < points.size(); i++)
    {
        Vector3d point = points[i];

        if (get_absolute_point_plane_distance(point, plane_params) > 1.5)
           continue;

        double nd = -normal.dot(point);
        double hd = -horizontal.dot(point);
        double vd = -vertical.dot(point);

        if (nd < min_n_d){
            min_n_pt = point;
            min_n_d = nd;
        }
        else if (nd > max_n_d){
            max_n_pt = point;
            max_n_d = nd;
        }

        if (hd < min_h_d){
            min_h_pt = point;
	        min_h_d = hd;
        }
        else if (hd > max_h_d){
            max_h_pt = point;
    	    max_h_d = hd;
	    }

        if (vd < min_v_d){
            min_v_pt = point;
	        min_v_d = vd;
	    }
        else if (vd > max_v_d){
            max_v_pt = point;
	        max_v_d = vd;
	    }	
    }

    vector<Vector4d> bound_planes;
    vector<Vector3d> bound_vertices;

    Vector4d left_plane;
    left_plane << horizontal, min_h_d;
    bound_planes.push_back(left_plane);

    Vector4d right_plane;
    right_plane << horizontal, max_h_d;
    bound_planes.push_back(right_plane);

    Vector4d front_plane;
    front_plane << normal, min_n_d;
    bound_planes.push_back(front_plane);

    Vector4d back_plane;
    back_plane << normal, max_n_d;
    bound_planes.push_back(back_plane);

    compute_vertices_from_planes(min_v_pt, bound_planes, bound_vertices);
    compute_vertices_from_planes(max_v_pt, bound_planes, bound_vertices);

    for (int i = 0; i < bound_vertices.size(); i++)
    {
        geometry_msgs::Point pt;
        pt.x = bound_vertices[i].x();
        pt.y = bound_vertices[i].y();
        pt.z = bound_vertices[i].z();

        vertices.push_back(pt);
        
        Vector3d t_pt(pt.x, 0.0, pt.z);
        if (t_pt.norm() > 500)
            return false;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    ROS_INFO("time taken for cuboid fitting is %g ms", elapsed_time_ms);

    return true;
}

/**
 * Vertices are in the following order
 * vertices 0-3 define the top face
 * vertices 4-7 define the bottom face
 * 
 * In each face, following order is used:
 * back right, front right, front left, back left
 **/
void create_cuboid_frame(vector<geometry_msgs::Point> &local_vertices, visualization_msgs::Marker &line_list, Isometry3d local2world)
{   
    vector<geometry_msgs::Point> vertices; 
    for (auto l_v: local_vertices)
    {
        Vector3d v(l_v.x, l_v.y, l_v.z);
        v = local2world * v.homogeneous();
        
        geometry_msgs::Point vertex;
        vertex.x = v[0];
        vertex.y = v[1];
        vertex.z = v[2];

        vertices.push_back(vertex);
    } 
    
    // Define the edges for top face
    line_list.points.push_back(vertices[0]);
    line_list.points.push_back(vertices[1]);

    line_list.points.push_back(vertices[1]);
    line_list.points.push_back(vertices[3]);

    line_list.points.push_back(vertices[3]);
    line_list.points.push_back(vertices[2]);

    line_list.points.push_back(vertices[2]);
    line_list.points.push_back(vertices[0]);

    // Define the edges for bottom face
    line_list.points.push_back(vertices[4]);
    line_list.points.push_back(vertices[5]);

    line_list.points.push_back(vertices[5]);
    line_list.points.push_back(vertices[7]);

    line_list.points.push_back(vertices[7]);
    line_list.points.push_back(vertices[6]);

    line_list.points.push_back(vertices[6]);
    line_list.points.push_back(vertices[4]);

    // Define the 4 edges connecting top face and bottom 
    line_list.points.push_back(vertices[0]);
    line_list.points.push_back(vertices[4]);

    line_list.points.push_back(vertices[1]);
    line_list.points.push_back(vertices[5]);

    line_list.points.push_back(vertices[2]);
    line_list.points.push_back(vertices[6]);

    line_list.points.push_back(vertices[3]);
    line_list.points.push_back(vertices[7]);
}

void cluster_plane_features(
    const sensor_msgs::PointCloudConstPtr &features_msg,
    cv::Mat &mask_img,
    Isometry3d world2local
)
{
    ROS_INFO("Point cloud has %d channels", (int)features_msg->channels.size());

    // Loop through all feature points
    for(int fi = 0; fi < features_msg->points.size(); fi++) {
        Vector3d fpoint;
        geometry_msgs::Point32 p = features_msg->points[fi];
        fpoint << p.x, p.y, p.z;
        Vector3d lpoint = world2local * fpoint;

        // if (lpoint.norm() > 100)
            // continue;

        int feature_id = (int)features_msg->channels[2].values[fi];

        if (mFeatures.find(feature_id) == mFeatures.end()) {
            int u = (int)features_msg->channels[0].values[fi];
            int v = (int)features_msg->channels[1].values[fi];

            Eigen::Matrix3d K;
            K << FOCAL_LENGTH, 0, COL/2,
                0, FOCAL_LENGTH, ROW/2,
                0, 0, 1;

            Vector3d pt = K * lpoint;
            pt /= pt[2];

            u = (int)pt.x();
            v = (int)pt.y();
            
            int plane_id = get_plane_id(u, v, mask_img);

            if ((plane_id != 0) && (plane_id != 39))// Ignore sky and ground points
            {
                if (! mPlaneFeatureIds.count(plane_id))
                {
                    Plane new_plane;
                    new_plane.plane_id = plane_id;
                    
                    mPlaneFeatureIds[plane_id] = new_plane;
                }
                mPlaneFeatureIds[plane_id].feature_ids.insert(feature_id);

                PlaneFeature new_plane_feature;
                new_plane_feature.plane_id = plane_id;
                mFeatures[feature_id] = new_plane_feature;
            }
        }

        mFeatures[feature_id].point = fpoint;
        mFeatures[feature_id].measurement_count++;
    }
}

map<int, Vector3d> draw_vp_lines(cv::Mat &gray_img, cv::Mat &mask, vector<Vector3d> &evps, vector<Vector3d> &normal_vectors)
{   
    auto t_start = std::chrono::high_resolution_clock::now();
    Eigen::Matrix3d K;
    K << FOCAL_LENGTH, 0, COL/2,
        0, FOCAL_LENGTH, ROW/2,
        0, 0, 1;

    map<int, int> plane_normal_ids;
    map<int, Vector3d> plane_normals;

    // show_img2.copyTo(tmp_img);
    // cv::cvtColor(mask_ptr_bgr8->image, tmp_img, CV_GRAY2RGB);
    std::vector<KeyLine> lines_klsd;
    cv::Mat lines_lsd_descr; 
    std::vector<cv::Point3d> vps(3);
    std::vector<std::vector<int> > clusters(3);
    std::vector<int> lines_vps;
    double f = K(0, 0);
    cv::Point2d pp(K(0, 2), K(1, 2));
    int LENGTH_THRESH = 5;
    
    ROS_INFO("Extracting line segments and vanishing points");
    extract_lines_and_vps(
        gray_img, 
        lines_klsd, lines_lsd_descr, 
        vps, clusters,
        lines_vps,
        f, pp, LENGTH_THRESH
    );

    ROS_INFO("Drawing vp quads");
    plane_normal_ids = drawVPQuads(gray_img, lines_klsd, clusters, mask);

    cv::Point3d normal0 = vps[0].cross(vps[1]);
    cv::Point3d normal1 = vps[2].cross(vps[1]);
    Vector3d normal_vector0(normal0.x, normal0.y, normal0.z);
    Vector3d normal_vector1(normal1.x, normal1.y, normal1.z);
    normal_vectors.push_back(normal_vector0);
    normal_vectors.push_back(normal_vector1);

    evps.push_back(Vector3d(vps[0].x, vps[0].y, vps[0].z));
    evps.push_back(Vector3d(vps[1].x, vps[1].y, vps[1].z));
    evps.push_back(Vector3d(vps[2].x, vps[2].y, vps[2].z));

    ROS_INFO("assigning plane normals");
    for (auto normal_id: plane_normal_ids)
	{
		if (
            (normal_id.first == 0)
            || (normal_id.first == 39)
        )
			continue;
		
        plane_normals[normal_id.first] = normal_vectors[normal_id.second].normalized();
	}

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    ROS_INFO("time taken for vanishing point computation is %g ms", elapsed_time_ms);

    return plane_normals;
}

geometry_msgs::Point toGeomPoint(Vector3d pt)
{
    geometry_msgs::Point point;
    point.x = pt.x();
    point.y = pt.y();
    point.z = pt.z();

    return point;
}

geometry_msgs::Point32 toGeomPoint32(Vector3d pt)
{
    geometry_msgs::Point32 point;
    point.x = pt.x();
    point.y = pt.y();
    point.z = pt.z();

    return point;
}

geometry_msgs::Point32 pointToPoint32(geometry_msgs::Point pt)
{
    geometry_msgs::Point32 pt32;
    pt32.x = pt.x;
    pt32.y = pt.y;
    pt32.z = pt.z;

    return pt32;
}

void write_normal_error(vector<Vector3d> vp_normals, vector<Vector3d> gt_normals)
{
    double error1 = 1 - std::max(
        fabs(vp_normals[0].dot(gt_normals[0]))
        , fabs(vp_normals[0].dot(gt_normals[1]))
    );

    double error2 = 1 - std::max(
        fabs(vp_normals[1].dot(gt_normals[0]))
        , fabs(vp_normals[1].dot(gt_normals[1]))
    );

    double error = (error1 + error2) / 2;

    ofstream file("vp_error.txt", std::ios_base::app);

    file << error << std::endl;

    file.close();
}

void write_estimated_normal_error(Vector3d estimated_normal, vector<Vector3d> gt_normals)
{
    double error = 1 - std::max(
        fabs(estimated_normal.dot(gt_normals[0]))
        , fabs(estimated_normal.dot(gt_normals[1]))
    );

    ofstream file("estim_error.txt", std::ios_base::app);

    file << error << std::endl;

    file.close();
}

void write_normal_and_distance_errors(Vector4d est_params, Vector4d gt_params, int plane_id, int est_num_of_points, int gt_num_of_points)
{
    ofstream file("plane_error.txt", std::ios_base::app);

    double normal_error = fabs(est_params.head<3>().normalized().dot(gt_params.head<3>().normalized()));
    double offset_error = fabs(fabs(est_params[3]) - fabs(gt_params[3]));

    file << plane_id << " " << normal_error << " " << offset_error << " " << fabs(gt_params[3]) << " " << est_num_of_points << " " << gt_num_of_points << std::endl;

    file.close();
}

vector<int> get_s_random_indices_within_n(int n /*range (1, n)*/, int s /*required number of samples*/)
{
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> uniform_distro(0, n); // define the range

    vector<int> rand_ints;

    for (int i = 0; i < s; i++)
    {
        rand_ints.push_back(uniform_distro(gen));
    }

    return rand_ints;
}

Vector4d fit_vertical_plane(vector<Vector3d> &plane_points)
{
    MatrixXd pts_mat(plane_points.size(), 3);
    Vector4d plane_params;

    for (int i = 0; i < (int)plane_points.size(); i++)
    {
        Vector3d c_pt = plane_points[i];

        Vector3d c_pt_flat(c_pt[0], c_pt[2], 1.0);

        pts_mat.row(i) = c_pt_flat.transpose();
    }

    Vector3d params;
    Eigen::JacobiSVD<MatrixXd> pt_svd(pts_mat, Eigen::ComputeFullU | Eigen::ComputeFullV);
    params = pt_svd.matrixV().col(pt_svd.matrixV().cols() - 1);

    plane_params << params[0], 0.0, params[1], params[2];
    plane_params /= plane_params.head<3>().norm();

    return plane_params;
}

double get_plane_inliers_error(vector<int> &inlier_indices, vector<Vector3d> &plane_points, Vector4d plane_model)
{
    double error = 0.0;
    for (int i = 0; i < inlier_indices.size(); i++)
    {
        error += get_absolute_point_plane_distance(plane_points[inlier_indices[i]], plane_model);
    }

    return error / inlier_indices.size();
}

Vector4d fit_vertical_plane_to_indices(vector<int> &indices, vector<Vector3d> &plane_points)
{
    vector<Vector3d> indexed_points;

    for(int i = 0; i < indices.size(); i++)
    {
        indexed_points.push_back(plane_points[indices[i]]);
    }

    return fit_vertical_plane(indexed_points);
}

/**
 * @brief implements a ransac for vertical plane
 * 
 * Formula to compute ransac number of iterations (N):
 * N = log(1-p)/log(1-((1-e)^s))
 * 
 * where:
 * p = desired probability that we get a good sample
 * s = number of points in a sample
 * e = probability that a point is outlier
 * 
 * @param plane_points 
 * @param plane_params 
 */
Vector4d fit_vertical_plane_ransac(vector<Vector3d> &plane_points, int plane_id)
{
    if (!mPlaneFeatureIds[plane_id].should_update)
        return mPlaneFeatureIds[plane_id].plane;

    auto t_start = std::chrono::high_resolution_clock::now();
    // Implement ransac for vertical planes
    // For each iteration:
    //      choose two random indices (as the plane is vertical, we just need two points)
    //      fit a vertical plane
    //      count the number of inliers
    //      if the count is greater than previous and if the error is also less than previous
    //      make current model as the best and save all inliers

    // Compute the number of iterations based on the outlier probability
    // Loop for 'n' iterations
    double p = 0.99; // p = desired probability that we get a good sample
    double s = 3; // s = number of points in a sample
    double e = 0.2; // e = probability that a point is outlier
    int N = (int)(log(1 - p) / log(1 - pow(1 - e, s)));
    N++;

    double plane_distance_threshold = 1.25;

    int bestNumOfInliers = 4;
    Vector4d bestFit;
    double bestError = 100000.0;

    for (int iter = 0; iter < N; iter++)
    {
        vector<int> maybeInliers = get_s_random_indices_within_n(plane_points.size(), s);
        Vector4d maybeModel = fit_vertical_plane_to_indices(maybeInliers, plane_points);

        vector<int> alsoInliers;
        for(int i = 0; i < plane_points.size(); i++)
        {
            if (find(maybeInliers.begin(), maybeInliers.end(), i) == maybeInliers.end()) // if not in maybeInliers
            {
                if (get_absolute_point_plane_distance(plane_points[i], maybeModel) <= plane_distance_threshold)
                {
                    alsoInliers.push_back(i);
                }
            }
        }

        // if ((alsoInliers.size() >= bestNumOfInliers))
        // {
            maybeInliers.insert(maybeInliers.end(), alsoInliers.begin(), alsoInliers.end());
            Vector4d betterModel = fit_vertical_plane_to_indices(maybeInliers, plane_points);
            double currentError = get_plane_inliers_error(maybeInliers, plane_points, betterModel);

            if (currentError < bestError)
            {
                bestFit = betterModel;
                bestError = currentError;
                bestNumOfInliers = maybeInliers.size();
            }
        // }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    ROS_INFO("time taken for RANSAC is %g ms", elapsed_time_ms);

    if (mPlaneFeatureIds[plane_id].best_num_of_inliers < bestNumOfInliers){
        mPlaneFeatureIds[plane_id].plane = bestFit;
        mPlaneFeatureIds[plane_id].best_num_of_inliers = bestNumOfInliers;
        mPlaneFeatureIds[plane_id].should_update = false;
    }

    return mPlaneFeatureIds[plane_id].plane;
}

map<int, vector<Vector3d>> cluster_depth_cloud(cv_bridge::CvImagePtr depth_ptr, cv_bridge::CvImagePtr mask_ptr, pcl::PointCloud<pcl::PointXYZRGB> &test_pcd, Isometry3d Ti, Isometry3d Tic)
{
    Eigen::Matrix3d K;
    K << FOCAL_LENGTH, 0, COL/2,
        0, FOCAL_LENGTH, ROW/2,
        0, 0, 1;
    
    cv::Mat raw_mask_img = mask_ptr->image;
    // cv::Mat mask_img = processMaskSegments(raw_mask_img);
    map<int, vector<Vector3d>> mDenseClusters;
    
    for (int i = 0; i < depth_ptr->image.rows; i=i+5)
    {
        for (int j = 0; j < depth_ptr->image.cols; j=j+5)
        {
            Vector3d cpt;
            Vector2d ipt(j, i);
            cpt = K.inverse() * ipt.homogeneous();
            cpt.normalize();
            float depth = depth_ptr->image.at<float>(i, j);
            if (std::isnan(depth) || depth > 50.0)
                continue;
            //std::cout << "Depth value is " << std::to_string(depth) << std::endl;
            cpt = depth * cpt;

            Vector3d w_pt;
            w_pt = Ti * (Tic * cpt);
            
            cv::Vec3b colors = raw_mask_img.at<cv::Vec3b>(i, j);

            int plane_id = color2id(colors[0], colors[1], colors[2]);
            
            pcl::PointXYZRGB pt;
            pt.x = w_pt.x();
            pt.y = w_pt.y();
            pt.z = w_pt.z();
            pt.r = colors[2];
            pt.g = colors[1];
            pt.b = colors[0];
            test_pcd.points.push_back(pt);

            if ((plane_id != 0) && (plane_id != 39))// Ignore sky and ground points
                mDenseClusters[plane_id].push_back(cpt);
        }
    }

    return mDenseClusters;
}

void update_global_point_cloud(
    const sensor_msgs::PointCloudConstPtr &features_msg,
    cv::Mat &mask_img,
    Isometry3d world2local
)
{
    ROS_INFO("Point cloud has %d channels", (int)features_msg->channels.size());

    map<int, Plane> mCurrentPlanes;
    map<int, int> mCurrentIdToPrevId;
    vector<int> common_plane_ids;

    map<int, PlaneFeature> mCurrentFeatures;

    std::map<unsigned long, int> color_index;

    for (int fid = 0; fid < features_msg->points.size(); fid++)
    {
        int feature_id = (int)features_msg->channels[2].values[fid];

        // Compute current id first
        Vector3d fpoint;
        geometry_msgs::Point32 p = features_msg->points[fid];
        fpoint << p.x, p.y, p.z;

        int u = (int)features_msg->channels[0].values[fid];
        int v = (int)features_msg->channels[1].values[fid];

        Vector3d lpoint = world2local * fpoint;
        Eigen::Matrix3d K;
        K << FOCAL_LENGTH, 0, COL/2,
            0, FOCAL_LENGTH, ROW/2,
            0, 0, 1;

        Vector3d pt = K * lpoint;
        pt /= pt[2];

        u = (int)pt.x();
        v = (int)pt.y();
        
        int current_plane_id = get_plane_id(u, v, mask_img);
        if (current_plane_id == 0)// || (current_plane_id == 39))
            continue;
            
        if (mCurrentPlanes.find(current_plane_id) == mCurrentPlanes.end())
        {
            Plane curr_plane;
            mCurrentPlanes[current_plane_id] = curr_plane;
        }

        if (mFeatures.find(feature_id) != mFeatures.end())
        {
            common_plane_ids.push_back(mFeatures[feature_id].plane_id);
        }

        mCurrentPlanes[current_plane_id].feature_ids.insert(feature_id);
        PlaneFeature new_feature;
        new_feature.point = fpoint;
        mCurrentFeatures[feature_id] = new_feature;
    }

    /**
     * @brief Find common points and associate current planes with existing planes
     * For each current plane:
     *      find common feature ids
     *      find an existing plane with maximum number of common features
     *      consider that as the mapping for current to previous plane
     */
    for (int c = 0; c < common_plane_ids.size(); c++)
    {
        int max_count = 0;
        int best_match_id = 0;

        for (auto &mCurrentPlane: mCurrentPlanes)
        {
            std::vector<int> common_feature_ids;
            Plane &commonPlane = mPlaneFeatureIds[common_plane_ids[c]];

            std::set_intersection(
                commonPlane.feature_ids.begin(), commonPlane.feature_ids.end(),
                mCurrentPlane.second.feature_ids.begin(), mCurrentPlane.second.feature_ids.end(),
                std::back_inserter(common_feature_ids)
            );

            if (common_feature_ids.size() > max_count){
                max_count = common_feature_ids.size();
                best_match_id = mCurrentPlane.first;
            }
        }

        if (max_count > 0 && best_match_id != 0)
            mCurrentIdToPrevId[best_match_id] = common_plane_ids[c];
    }

    /**
     * feature id is not there in the current map
     * there are three cases for this feature point
     * case 1: it belongs to already mapped planes
     * case 2: it belongs to a new plane
     * case 3: it is invalid (not inside any mask)
     * 
     * feature belongs to a existing plane
     * this should be used to map current plane id with existing plane id
     */
    for(auto mCPF: mCurrentPlanes) {
        int current_plane_id = mCPF.first;

        // one of the features in current cloud belongs to existing planes
        if (mCurrentIdToPrevId.find(current_plane_id) != mCurrentIdToPrevId.end())
        {
            int previous_plane_id = mCurrentIdToPrevId[current_plane_id];
            if ((previous_plane_id == 0) || (previous_plane_id == 39))
                continue;

            for (auto &feature_id: mCPF.second.feature_ids) 
            {   
                //int feature_id = mCPF.second.feature_ids[cfid];

                // add this feature to existing plane
                if (mFeatures.find(feature_id) == mFeatures.end()) 
                {
                    mPlaneFeatureIds[previous_plane_id].feature_ids.insert(feature_id);

                    PlaneFeature new_plane_feature;
                    new_plane_feature.plane_id = previous_plane_id;
                    mFeatures[feature_id] = new_plane_feature;
                }
                else if (mFeatures[feature_id].plane_id != previous_plane_id) {
                    mFeatures[feature_id].is_outlier = true;
                    continue;
                }

                mFeatures[feature_id].point = mCurrentFeatures[feature_id].point;
                mFeatures[feature_id].measurement_count++;
                mPlaneFeatureIds[mFeatures[feature_id].plane_id].should_update = true;
            }
        }
        else // these cluster of features in current frame belong to a new plane
        {
            int new_plane_id = plane_id_counter;
            plane_id_counter++;
            //for (int cfid = 0; cfid < mCPF.second.feature_ids.size(); cfid++) 
            for (auto &feature_id: mCPF.second.feature_ids) 
            {   
                //int feature_id = mCPF.second.feature_ids[cfid];

                Plane new_plane;
                new_plane.plane_id = new_plane_id;
                    
                mPlaneFeatureIds[new_plane_id] = new_plane;

                // add this feature to existing plane
                if (mFeatures.find(feature_id) == mFeatures.end())
                {
                    mPlaneFeatureIds[new_plane_id].feature_ids.insert(feature_id);

                    PlaneFeature new_plane_feature;
                    new_plane_feature.plane_id = new_plane_id;
                    mFeatures[feature_id] = new_plane_feature;
                }

                mFeatures[feature_id].point = mCurrentFeatures[feature_id].point;
                mFeatures[feature_id].measurement_count++;
                mPlaneFeatureIds[mFeatures[feature_id].plane_id].should_update = true;
            }
        }
    }
}
