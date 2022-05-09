#include "local_map.h"

LocalMap::LocalMap(sensor_msgs::PointCloudConstPtr features, nav_msgs::OdometryConstPtr odometry, sensor_msgs::ImageConstPtr mask)
{
    features_msg = features;
    odometry_msg = odometry;
    mask_msg = mask;

    process_odometry();
    ROS_INFO("Received features, odometry and mask");
}

void LocalMap::process_odometry()
{
    Tic.linear() = RIC[0];
    Tic.translation() = TIC[0];
    ROS_INFO("Computed imu-camera transform");

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

    Ti.linear() = quat.normalized().toRotationMatrix();
    Ti.translation() = trans;
    
    world2local = Tic.inverse() * Ti.inverse();

    ROS_INFO("Computed world to local transform");
    
    // set colour id to 0
    color_index[(unsigned long)0] = 0;
}

void LocalMap::cluster_points()
{
    ROS_INFO("Point cloud has %d channels and %d points", (int)features_msg->channels.size(), (int)features_msg->points.size());

    cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat mask_img = mask_ptr->image;
    
    // Compute world to local transform
    Isometry3d world2local = Tic.inverse() * Ti.inverse(); 

    for (int fid = 0; fid < features_msg->points.size(); fid++)
    {
        int feature_id = (int)features_msg->channels[2].values[fid];

        // Compute plane id
        Vector3d fpoint;
        geometry_msgs::Point32 p = features_msg->points[fid];
        fpoint << p.x, p.y, p.z;

        int u = (int)features_msg->channels[0].values[fid];
        int v = (int)features_msg->channels[1].values[fid];

        Vector3d lpoint = world2local * fpoint;
        //Vector3d cpoint(lpoint.x(), 0.0, lpoint.z());
        //if (cpoint.norm() > 50)
        //    continue;

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
            if (! mPlanes.count(plane_id))
            {
                Plane new_plane;
                new_plane.plane_id = plane_id;
                
                mPlanes[plane_id] = new_plane;
            }
            mPlanes[plane_id].feature_ids.insert(feature_id);

            if (! mPlaneFeatures.count(feature_id))
            {
                PlaneFeature new_plane_feature;
                mPlaneFeatures[feature_id] = new_plane_feature;
            }
            mPlaneFeatures[feature_id].point = fpoint;
            mPlaneFeatures[feature_id].plane_id = plane_id;
            mPlaneFeatures[feature_id].measurement_count++;
        }
    }

    ROS_INFO("Found %d planes", (int)mPlanes.size());
}

void LocalMap::fit_cuboids()
{
    ROS_INFO("Fitting cuboids");
    //For each segmented plane
    for (auto& iter_plane: mPlanes)
    {   
        int plane_id = iter_plane.first;
        
	ROS_INFO("Fitting cuboid to plane %d ", plane_id);
        // Compute color of this plane cluster
        // unsigned long hex = id2color(plane_id);
        int r = 255;//((hex >> 16) & 0xFF);
        int g = 255;//((hex >> 8) & 0xFF);
        int b = 0;//((hex) & 0xFF);

        pcl::PointCloud<pcl::PointXYZRGB> plane_pcd;
        pcl::PointCloud<pcl::PointXYZRGB> filtered_plane_pcd;
        
        ROS_INFO("Number of features in plane id %d are %d", iter_plane.first, (int)iter_plane.second.feature_ids.size());
        // Create a coloured point cloud
        for (auto feature_id: iter_plane.second.feature_ids)
        {
            if (mPlaneFeatures[feature_id].measurement_count < 3)
                continue;
            Vector3d w_pt = mPlaneFeatures[feature_id].point;
            pcl::PointXYZRGB pt;
            pt.x = w_pt.x();
            pt.y = w_pt.y();
            pt.z = w_pt.z();
            pt.r = b;
            pt.g = g;
            pt.b = r;
            plane_pcd.points.push_back(pt);
        }
        
        // Create the filtering object
        plane_pcd.is_dense = false;
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
        ror.setInputCloud (plane_pcd.makeShared());
        ror.setRadiusSearch (3);
        ror.setMinNeighborsInRadius (2);
        ror.setKeepOrganized (true);
        ror.filter (filtered_plane_pcd);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(filtered_plane_pcd, filtered_plane_pcd, indices);
        
        ROS_INFO("Before filtering: %d, After filtering: %d", (int)plane_pcd.points.size(), (int)filtered_plane_pcd.points.size());
        //std::cout << "Before" << std::endl;
        //std::cout << plane_pcd << std::endl;
        //std::cout << "After" << std::endl;
        //std::cout << filtered_plane_pcd << std::endl;

        if (filtered_plane_pcd.points.size() < 10)
            continue;
        
        vector<Vector3d> plane_points;
        for (int pid = 0; pid < filtered_plane_pcd.points.size(); pid++)
        {   
            pcl::PointXYZRGB pt = filtered_plane_pcd.points[pid];
            Vector3d w_pt(pt.x, pt.y, pt.z);
            Vector3d c_pt = Tic.inverse() * (Ti.inverse() * w_pt);
                
            plane_points.push_back(c_pt);
        }

        Vector4d params = fit_vertical_plane_ransac(plane_points);
        mPlanes[plane_id].params = params;
        ROS_INFO("Plane params %lf, %lf, %lf, %lf", params[0], params[1], params[2], params[3]);

        vector<geometry_msgs::Point> vertices;
        fit_cuboid_to_points(params, plane_points, vertices);
        mPlanes[plane_id].vertices = vertices;
        mPlanes[plane_id].is_initialized = true;
    }
}

double LocalMap::compute_plane_merging_cost(Plane curr_plane, Plane prev_plane, Isometry3d w2l)
{
    // Compute plane-incidence error of prev_plane points
    vector<Vector3d> prev_plane_points;
    
    for (auto &feature_id: prev_plane.feature_ids)
    {
        prev_plane_points.push_back(w2l * mPlaneFeatures[feature_id].point);
    }

    return get_plane_points_error(prev_plane_points, curr_plane.params);
}

void LocalMap::merge_old_map(LocalMap old_map)
{
    for (auto& plane: mPlanes)
    {
        // Find matching plane
        int max_count = 1;
        int best_match_id = 0;

        for (auto &old_plane: old_map.mPlanes)
        {
            std::vector<int> common_feature_ids;

            std::set_intersection(
                plane.second.feature_ids.begin(), plane.second.feature_ids.end(),
                old_plane.second.feature_ids.begin(), old_plane.second.feature_ids.end(),
                std::back_inserter(common_feature_ids)
            );

            if (common_feature_ids.size() > max_count){
                max_count = common_feature_ids.size();
                best_match_id = old_plane.first;
            }
        }

        if (max_count > 0 && best_match_id != 0)
        {
            // see if planes can be merged
            if (compute_plane_merging_cost(old_map.mPlanes[best_match_id], plane.second, old_map.world2local) < 3)
            {
                // Merge planes
                Plane matching_plane = old_map.mPlanes[best_match_id];
		ROS_INFO("Before merging %d features", (int)mPlanes[plane.first].feature_ids.size());
                mPlanes[plane.first].feature_ids.insert(matching_plane.feature_ids.begin(), matching_plane.feature_ids.end());
                old_map.mPlanes.erase(best_match_id);
		ROS_INFO("After merging %d features", (int)mPlanes[plane.first].feature_ids.size());
                ROS_INFO("********* MERGED planes ********");
            }
        }
    }
    //ROS_INFO("Before inserting %d planes", (int)mPlanes.size());
    //mPlanes.insert(old_map.mPlanes.begin(), old_map.mPlanes.end());
    //ROS_INFO("After inserting %d planes", (int)mPlanes.size());
    
    //ROS_INFO("Before inserting %d colours", (int)color_index.size());
    //for (auto it = color_index.begin(); it != color_index.end(); ++it) {
//	ROS_INFO(" %d ", (int)it->second);
 //   }
    
  //  color_index.insert(old_map.color_index.begin(), old_map.color_index.end());
   // ROS_INFO("After inserting %d colours", (int)color_index.size());
    //for (auto it = old_map.color_index.begin(); it != old_map.color_index.end(); ++it) {
//	ROS_INFO(" %d ", (int)it->second);
 //   }
}

void LocalMap::merge_old_map2(LocalMap old_map)
{
    for (auto &old_plane: old_map.mPlanes)
    {
        // Find matching plane
        int max_count = 0;
        int best_match_id = 0;

        for (auto &plane: mPlanes)
        {
            std::vector<int> common_feature_ids;

            std::set_intersection(
                plane.second.feature_ids.begin(), plane.second.feature_ids.end(),
                old_plane.second.feature_ids.begin(), old_plane.second.feature_ids.end(),
                std::back_inserter(common_feature_ids)
            );

            if (common_feature_ids.size() > max_count){
                max_count = common_feature_ids.size();
                best_match_id = old_plane.first;
            }
        }

        if (max_count > 0 && best_match_id != 0)
        {
            // see if planes can be merged
            if (compute_plane_merging_cost(old_map.mPlanes[best_match_id], plane.second, old_map.world2local) < 3)
            {
                // Merge planes
                Plane matching_plane = old_map.mPlanes[best_match_id];
		ROS_INFO("Before merging %d features", (int)mPlanes[plane.first].feature_ids.size());
                mPlanes[plane.first].feature_ids.insert(matching_plane.feature_ids.begin(), matching_plane.feature_ids.end());
                old_map.mPlanes.erase(best_match_id);
		ROS_INFO("After merging %d features", (int)mPlanes[plane.first].feature_ids.size());
                ROS_INFO("********* MERGED planes ********");
            }
        }
    }
    //ROS_INFO("Before inserting %d planes", (int)mPlanes.size());
    //mPlanes.insert(old_map.mPlanes.begin(), old_map.mPlanes.end());
    //ROS_INFO("After inserting %d planes", (int)mPlanes.size());
    
    //ROS_INFO("Before inserting %d colours", (int)color_index.size());
    //for (auto it = color_index.begin(); it != color_index.end(); ++it) {
//	ROS_INFO(" %d ", (int)it->second);
 //   }
    
  //  color_index.insert(old_map.color_index.begin(), old_map.color_index.end());
   // ROS_INFO("After inserting %d colours", (int)color_index.size());
    //for (auto it = old_map.color_index.begin(); it != old_map.color_index.end(); ++it) {
//	ROS_INFO(" %d ", (int)it->second);
 //   }
}



void LocalMap::publish_cuboids(ros::Publisher cuboids_pub, ros::Publisher vertices_pub)
{
    ROS_INFO("Publishing cuboids");
    sensor_msgs::PointCloud vertices_cloud;
    vertices_cloud.header = features_msg->header;
    sensor_msgs::ChannelFloat32 plane_id_ch;

    visualization_msgs::Marker line_list;
    line_list.header = odometry_msg->header;

    // line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;

    line_list.id = id;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    // LINE_LIST markers use only the x component of scale, for the line width
    line_list.scale.x = 0.08;

    // Line list is green
    // if (fabs(normal[0]) > fabs(normal[2]))
    line_list.color.r = 1.0;
    // else
        // line_list.color.b = 1.0;
    line_list.color.a = 1.0;


    for (auto& iter_plane: mPlanes)
    {
        if (! iter_plane.second.is_initialized)
            continue;
        create_cuboid_frame(iter_plane.second.vertices, line_list, (Ti * Tic));
        for (int vid = 0; vid < iter_plane.second.vertices.size(); vid++)
        {
            vertices_cloud.points.push_back(pointToPoint32(iter_plane.second.vertices[vid]));
            plane_id_ch.values.push_back(iter_plane.first);
        }
    }

    cuboids_pub.publish(line_list);

    vertices_cloud.channels.push_back(plane_id_ch);
    vertices_pub.publish(vertices_cloud);
}

void LocalMap::publish_clusters(ros::Publisher clusters_pub)
{
    ROS_INFO("Publishing clusters");
    pcl::PointCloud<pcl::PointXYZRGB> clusters_pcd;
    
    // For each segmented plane
    for (auto& iter_plane: mPlanes)
    {   
        int plane_id = iter_plane.first;
        ROS_INFO("Number of features in plane id %d are %d", iter_plane.first, (int)iter_plane.second.feature_ids.size());
        
        // Compute color of this plane cluster
        // unsigned long hex = id2color(plane_id);
        int r = 255; //((hex >> 16) & 0xFF);
        int g = 255; //((hex >> 8) & 0xFF);
        int b = 0; //((hex) & 0xFF);

        pcl::PointCloud<pcl::PointXYZRGB> plane_pcd;
        // Create a coloured point cloud
        for (auto feature_id: iter_plane.second.feature_ids)
        {
            if (mPlaneFeatures[feature_id].measurement_count < 3)
                continue;
            Vector3d w_pt = mPlaneFeatures[feature_id].point;
            pcl::PointXYZRGB pt;
            pt.x = w_pt.x();
            pt.y = w_pt.y();
            pt.z = w_pt.z();
            pt.r = b;
            pt.g = g;
            pt.b = r;
            plane_pcd.points.push_back(pt);
        }

        // Create the filtering object
        plane_pcd.is_dense = false;
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
        ror.setInputCloud (plane_pcd.makeShared());
        ror.setRadiusSearch (3);
        ror.setMinNeighborsInRadius (2);
        ror.setKeepOrganized (true);
        ror.filter (plane_pcd);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(plane_pcd, plane_pcd, indices);
        
        if (plane_pcd.points.size() < 10)
            continue;
        clusters_pcd += plane_pcd;
    }

    // Convert to ROS PointCloud2 and publish
    sensor_msgs::PointCloud2 clusters_cloud;
    pcl::toROSMsg(clusters_pcd, clusters_cloud);
    clusters_cloud.header = features_msg->header;
    clusters_pub.publish(clusters_cloud);
}
