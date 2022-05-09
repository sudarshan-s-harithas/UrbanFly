#ifndef _LOCAL_MAP_H_
#define _LOCAL_MAP_H_
#pragma once

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
#include <pcl/filters/filter.h>

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

using namespace std;
using namespace Eigen;

struct Plane
{
    Vector4d params;
    vector<geometry_msgs::Point> vertices;
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

class LocalMap
{
public:
    LocalMap(){}
    LocalMap(sensor_msgs::PointCloudConstPtr features_msg, nav_msgs::OdometryConstPtr odometry_msg, sensor_msgs::ImageConstPtr mask_msg);
    
    // Computes the transforms from odometry
    void process_odometry();
    // Clusters current point cloud based on the mask image
    void cluster_points();
    void filter_clusters();
    void fit_cuboids();
    void merge_old_map(LocalMap map);
    void merge_old_map2(LocalMap map);
    double compute_plane_merging_cost(Plane plane1, Plane plane2, Isometry3d w2l);
    void publish_clusters(ros::Publisher clusters_pub);
    void publish_cuboids(ros::Publisher cuboids_pub, ros::Publisher vertices_pub);

    // ID of the local map
    int id = 2;

    // Messages
    sensor_msgs::PointCloudConstPtr features_msg;
    nav_msgs::OdometryConstPtr odometry_msg;
    sensor_msgs::ImageConstPtr mask_msg;

    // Odometry
    Isometry3d Tic;
    Isometry3d Ti;
    Isometry3d world2local;

    // Clusters
    map<int, Plane> mPlanes;
    map<int, PlaneFeature> mPlaneFeatures;

    // RViz markers
    pcl::PointCloud<pcl::PointXYZRGB> plane_pcd;

    // Misc
    int plane_counter;
    
    // Each color is stored in the hex format
    std::map<unsigned long, int> color_index;

private:
    // Returns the id from color specified by (r, g, b) values
    int color2id(int r, int g, int b)
    {
        // TODO: Handle the ground plane in the mapping
        unsigned long hex = ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff);

        if (color_index.find(hex) == color_index.end()) {
            color_index[hex] = plane_counter++;//1000 + mPlanes.size();//1000 + (int)color_index.size();
        }

        // std::cout << "Queried for color " << std::to_string(r) << " " << std::to_string(g) << " " << std::to_string(b) << std::endl;
        // std::cout << "ID is " << std::to_string(color_index[hex]) << std::endl;
        return color_index[hex];
    }

    // Returns the (r, g, b) values of a color in hex format
    unsigned long id2color(int id)
    {
        for (auto it = color_index.begin(); it != color_index.end(); ++it) {
            if (it->second == id)
                return it->first;   
        }
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

double get_plane_points_error(vector<Vector3d> &plane_points, Vector4d plane_params)
{
    double error = 0.0;
    for (int i = 0; i < plane_points.size(); i++)
    {
        error += get_absolute_point_plane_distance(plane_points[i], plane_params);
    }

    return error / plane_points.size();
}

double get_plane_inliers_error(vector<int> &inlier_indices, vector<Vector3d> &plane_points, Vector4d plane_model)
{
    vector<Vector3d> inlier_points;
    
    for (int i = 0; i < inlier_indices.size(); i++)
    {
        inlier_points.push_back(plane_points[inlier_indices[i]]);
    }

    return get_plane_points_error(inlier_points, plane_model);
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
Vector4d fit_vertical_plane_ransac(vector<Vector3d> &plane_points)
{
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

        if ((alsoInliers.size() >= bestNumOfInliers))
        {
            maybeInliers.insert(maybeInliers.end(), alsoInliers.begin(), alsoInliers.end());
            Vector4d betterModel = fit_vertical_plane_to_indices(maybeInliers, plane_points);
            double currentError = get_plane_inliers_error(maybeInliers, plane_points, betterModel);

            if (currentError < bestError)
            {
                bestFit = betterModel;
                bestError = currentError;
                bestNumOfInliers = maybeInliers.size();
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    ROS_INFO("time taken for RANSAC is %g ms", elapsed_time_ms);

    return bestFit;
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

bool fit_cuboid_to_points(Vector4d plane_params, vector<Vector3d> points, vector<geometry_msgs::Point> &vertices)
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

        if (get_absolute_point_plane_distance(point, plane_params) > 1.25)
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
        
        //Vector3d t_pt(pt.x, 0.0, pt.z);
        //if (t_pt.norm() > 500)
        //    return false;
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

geometry_msgs::Point32 pointToPoint32(geometry_msgs::Point pt)
{
    geometry_msgs::Point32 pt32;
    pt32.x = pt.x;
    pt32.y = pt.y;
    pt32.z = pt.z;

    return pt32;
}

};

#endif //_LOCAL_MAP_H_
