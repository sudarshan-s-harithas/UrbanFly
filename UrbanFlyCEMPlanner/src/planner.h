#pragma once

/**
 * To subscribe to
 *  poses
 *  plane segment params
 * 
 * Maintain header wise 
 *  poses
 *  plane segment params
 * 
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

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>
#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/PointCloud.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "eigenmvn.h"


#include "objects.h"


// #include "MMD.h"

using namespace std;
using namespace Eigen;

ros::Publisher pub_paths;
ros::Publisher pub_paths2;
ros::Publisher pub_colliding_cloud;
ros::Publisher pub_free_cloud;

// MMDFunctions::MMD_variants MMDF;
std::vector<double> means;
std::vector<double> covariances;
std::vector<double> weights;

map<double, vector<Vector4d>> plane_measurements;

MatrixXd diff(MatrixXd input)
{
  MatrixXd output;
  output = MatrixXd::Zero(input.rows()-1, input.cols());

  output = input.block(1, 0, input.rows()-1, input.cols()) - input.block(0, 0, input.rows()-1, input.cols());

  return output;
}

void read_vector(std::string file_path, std::vector<double> &output)
{
    std::cout << "reading values from file " << file_path << std::endl;
    std::ifstream file(file_path);
    std::string line;
    // std::vector<double> output;

    std::getline(file, line);
    std::stringstream line_stream(line);

    double value;
    while(line_stream >> value)
        output.push_back(value);

    file.close();
    ROS_INFO("Read vector file with %d values", output.size());
    // return output;
}

void print_vector(std::string title, std::vector<double> input)
{
    std::cout << title << ": ";
    for (int i = 0; i < input.size(); i++)
    {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
}

Eigen::MatrixXf sample_n_values(int n)
{
    // std::ofstream samples_file("samples.txt");
    std::random_device rd{};
    std::mt19937 gen{rd()};

    using normal_dist   = std::normal_distribution<>;
    using discrete_dist = std::discrete_distribution<std::size_t>;

    auto G = std::array<normal_dist, 4>{
        normal_dist{means[0], covariances[0]}, // mean, stddev of G[0]
        normal_dist{means[1], covariances[1]}, // mean, stddev of G[1]
        normal_dist{means[2], covariances[2]} , // mean, stddev of G[2]
        normal_dist{means[3], covariances[3]}  // mean, stddev of G[3]
    };

    auto w = discrete_dist{
        weights[0], // weight of G[0]
        weights[1], // weight of G[1]
        weights[2],  // weight of G[2]
        weights[3]  // weight of G[2]
    };

    Eigen::MatrixXf radius(1, n);  
    radius.setOnes();
    
    // vector<double> samples;
    Eigen::MatrixXf samples(1, n);
    for (int i = 0 ; i < n; i++){
        auto index = w(gen);
        auto temp_noise_val = G[index](gen);

        // samples_file << std::to_string(temp_noise_val) << std::endl;
        // samples.push_back(temp_noise_val);
        samples(0, i) = radius(0, i) - (float) temp_noise_val;
    }
    // samples_file.close();
    // std::cout << "noise values are written to file" << std::endl;
    return samples;
}

void init_gmm_values(std::string SRC_PATH)
{   
  ROS_INFO("initializing gmm params");
  read_vector(SRC_PATH + "/GMM_4_means.txt", means);
  read_vector(SRC_PATH + "/GMM_4_covariances.txt", covariances);
  read_vector(SRC_PATH + "/GMM_4_weights.txt", weights);

  ROS_INFO("Read %d means", (int)means.size());
  ROS_INFO("Read %d covariances", (int)covariances.size());
  ROS_INFO("Read %d weights", (int)weights.size());
}

// Implement MMD cost
double getMMDcost(double sdf_value)
{ 
  int n = 100;
  // ROS_INFO("Computing MMD cost for sdf value %g", sdf_value);
  // Sample 100 noise values from GMM distro
  Eigen::MatrixXf samples = sample_n_values(n);
  Eigen::MatrixXf zeros_row(1, n);
  zeros_row.setZero();

  Eigen::MatrixXf actual_distance = Eigen::MatrixXf::Constant( 1, n, sdf_value);
  samples = zeros_row.cwiseMax( samples - actual_distance );

  return 100 ; 

  // return (double) MMDF.MMD_transformed_features(samples);
}

geometry_msgs::Point32 pointToPoint32(geometry_msgs::Point pt)
{
    geometry_msgs::Point32 pt32;
    pt32.x = pt.x;
    pt32.y = pt.y;
    pt32.z = pt.z;

    return pt32;
}