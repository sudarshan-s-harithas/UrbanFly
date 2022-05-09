#include "mapper.h"

sensor_msgs::PointField getFieldWithName(string name)
{
    sensor_msgs::PointField pf;
    pf.name = name;

    return pf;
}

void depth_image_callback(
        const sensor_msgs::ImageConstPtr &depth_msg, 
        const sensor_msgs::ImageConstPtr &mask_msg,
        const nav_msgs::OdometryConstPtr &odometry_msg
)
{
    sensor_msgs::PointCloud2 test_cloud;

    pcl::PointCloud<pcl::PointXYZRGB> test_pcd;

    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(depth_msg);
    cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(mask_msg);

    // std::cout << "Encoding of depth image is " << depth_msg->encoding << std::endl;

    Isometry3d Tic;
    Tic.linear() = RIC[0];
    Tic.translation() = TIC[0];

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

    map<int, vector<Vector3d>> dense_clusters = cluster_depth_cloud(depth_ptr, mask_ptr, test_pcd, Ti, Tic);

    pcl::toROSMsg(test_pcd, test_cloud);
    test_cloud.header = odometry_msg->header;
    dense_pub.publish(test_cloud);

    return;
    /**
    sensor_msgs::PointCloud frame_cloud;
    frame_cloud.header = odometry_msg->header;
    sensor_msgs::ChannelFloat32 plane_id_ch;
    
    visualization_msgs::MarkerArray ma;

    Vector3d vertical(0, 1, 0);

    if (dense_clusters.size() == 0)
        return;

    visualization_msgs::Marker line_list;
    line_list.header = odometry_msg->header;
    line_list.pose.orientation.w = 1.0;

    line_list.id = 2;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    line_list.scale.x = 0.08;

    line_list.color.r = 1.0;
    line_list.color.a = 1.0;

    // Print number of features per plane
    for (auto & cluster: dense_clusters)
    {
        ROS_INFO("Number of features in plane id %d are %d", cluster.first, (int)cluster.second.size());

        int max_points = min((int)2000, (int)cluster.second.size());
        vector<Vector3d> plane_points;

        for (int c = 0; c < max_points; c++)
        {
            plane_points.push_back(cluster.second[c]);    
        }

        if (plane_points.size() < 5)
            continue;

        vector<geometry_msgs::Point> vertices;

        Vector3d normal;
        double d = 0.0;

        Vector4d params;
        params = fit_vertical_plane_ransac(plane_points);
        normal = params.head<3>();
        d = params[3];

        if ((normal.norm() > 0.001) && (fabs(params[3]) > 0.001))
        {
            Vector4d normed_params(normal[0], normal[1], normal[2], d);

            if (fabs(normal.dot(vertical)) < 0.5)
            {             
                vector<Vector3d> vp_normals;
                if (fit_cuboid_to_point_cloud(normed_params, plane_points, vertices, vp_normals))
                {
                    create_cuboid_frame(vertices, line_list, (Ti * Tic));

                    for (int vid = 0; vid < vertices.size(); vid++)
                    {
                        frame_cloud.points.push_back(pointToPoint32(vertices[vid]));
                        plane_id_ch.values.push_back(cluster.first);
                    }
                }
                else {
                    ROS_INFO("Not plotting cuboid %d with params %f %f %f %f", cluster.first, normed_params[0], normed_params[1], normed_params[2], normed_params[3]);
                }
            }
        }
    }

    marker_pub.publish(line_list);

    // Process a particular plane id
    frame_cloud.channels.push_back(plane_id_ch);
    frame_pub.publish(frame_cloud);**/
}

void mapping_callback(
    const sensor_msgs::PointCloudConstPtr &features_msg,
    const nav_msgs::OdometryConstPtr &odometry_msg,
    const sensor_msgs::ImageConstPtr &img_msg,
    const sensor_msgs::ImageConstPtr &mask_msg
)
{
    Vector3d ggoal(25.0, -5.0, 2.5); 

    ROS_INFO("Image message timestamp %f", img_msg->header.stamp.toSec());
    ROS_INFO("Features message timestamp %f", features_msg->header.stamp.toSec());
    ROS_INFO("Mask message timestamp %f\n", mask_msg->header.stamp.toSec());

    // Step 1: Cluster all the feature points based on their plane ids
    cv_bridge::CvImagePtr mask_ptr = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat raw_mask_img = mask_ptr->image;
    
    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img = img_ptr->image;
    ROS_INFO("Received point cloud with %d points", (int)features_msg->points.size());
    ROS_INFO("Received image with dimensions (%d, %d)", img.rows, img.cols);

    sensor_msgs::PointCloud frame_cloud;
    frame_cloud.header = features_msg->header;
    sensor_msgs::ChannelFloat32 plane_id_ch;
    sensor_msgs::PointCloud2 test_cloud;
    sensor_msgs::PointCloud2 test_cloud2;
    pcl::PointCloud<pcl::PointXYZRGB> test_pcd;
    pcl::PointCloud<pcl::PointXYZRGB> test_pcd2;

    vector<int> plane_ids;

    Isometry3d Tic;
    Tic.linear() = RIC[0];
    Tic.translation() = TIC[0];

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

    cv::Mat mask_img = processMaskSegments(raw_mask_img);
    update_global_point_cloud(features_msg, mask_img, Tic.inverse() * Ti.inverse());

    ROS_INFO("Clustered the feature points based on %d planes", (int)mPlaneFeatureIds.size());

    Vector3d lgoal = Tic.inverse() * (Ti.inverse() * ggoal);
    lgoal[1] = 0.0;
    sensor_msgs::PointCloud goal_pcd;

    goal_pcd.header = features_msg->header;
    goal_pcd.points.push_back(toGeomPoint32(lgoal));
    lgoal_pub.publish(goal_pcd);

    visualization_msgs::MarkerArray ma;

    Vector3d vertical(0, 1, 0);

    if (mPlaneFeatureIds.size() == 0)
        return;
        
    // ROS_INFO("Drawing quads for %d planes", plane_ids.size());
    // draw_quads(img, mask_img, plane_ids);
    // cv::Mat gray_img;
    // cv::cvtColor(img, gray_img, CV_BGR2GRAY);

    vector<Vector3d> vps;
    vector<Vector3d> vp_normals1;
    map<int, Vector3d> normals_map = draw_vp_lines(img, mask_img, vps, vp_normals1);

    vector<Vector3d> vp_normals;
    Vector3d x_axis(1, 0, 0);
    Vector3d y_axis(0, 1, 0);
    Vector3d lx_axis = (Tic.linear().inverse() * (Ti.linear().inverse() * y_axis).normalized()).normalized();
    Vector3d ly_axis = lx_axis.cross(vertical).normalized();
    vp_normals.push_back(lx_axis);
    vp_normals.push_back(ly_axis);

    // write_normal_error(vp_normals1, vp_normals);

    visualization_msgs::Marker line_list;
    line_list.header = odometry_msg->header;

    // line_list.action = visualization_msgs::Marker::ADD;
    line_list.pose.orientation.w = 1.0;

    line_list.id = 2;
    line_list.type = visualization_msgs::Marker::LINE_LIST;

    // LINE_LIST markers use only the x component of scale, for the line width
    line_list.scale.x = 0.08;

    // Line list is green
    // if (fabs(normal[0]) > fabs(normal[2]))
    line_list.color.r = 1.0;
    // else
        // line_list.color.b = 1.0;
    line_list.color.a = 1.0;

    // Print number of features per plane
    for (auto const& sFeatureIds: mPlaneFeatureIds)
    {
        int plane_id = sFeatureIds.first;
        ROS_INFO("Number of features in plane id %d are %d", sFeatureIds.first, (int)sFeatureIds.second.feature_ids.size());

        vector<Vector3d> plane_points;
        
        pcl::PointCloud<pcl::PointXYZRGB> plane_pcd;
        for (auto feature_id: sFeatureIds.second.feature_ids)
        {
            Vector3d w_pt = mFeatures[feature_id].point;

            Vector3d c_pt = Tic.inverse() * (Ti.inverse() * w_pt);
            
            Vector3d t_pt(c_pt[0], 0.0, c_pt[2]);
            if (mFeatures[feature_id].measurement_count > 1)
            {
                // unsigned long hex = id2color(plane_id);
                int r = 255;//((hex >> 16) & 0xFF);
                int g = 255;//((hex >> 8) & 0xFF);
                int b = 0;//((hex) & 0xFF);

                pcl::PointXYZRGB pt;
                pt.x = w_pt.x();
                pt.y = w_pt.y();
                pt.z = w_pt.z();
                pt.r = b;
                pt.g = g;
                pt.b = r;
                plane_pcd.points.push_back(pt);
            }
        }

        // Create the filtering object
        pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
        ror.setInputCloud (plane_pcd.makeShared());
        ror.setRadiusSearch (1);
        ror.setMinNeighborsInRadius (5);
        // ror.setKeepOrganized (true);
        ror.filter (plane_pcd);
        
        if (plane_pcd.points.size() < 10)
            continue;

        for (int pid = 0; pid < plane_pcd.points.size(); pid++)
        {   
            pcl::PointXYZRGB pt = plane_pcd.points[pid];
            Vector3d w_pt(pt.x, pt.y, pt.z);

            Vector3d c_pt = Tic.inverse() * (Ti.inverse() * w_pt);
            
            // Vector3d t_pt(c_pt[0], 0.0, c_pt[2]);
            // if ((t_pt.norm() <= 35) && (c_pt.norm() > 2) && (mFeatures[feature_id].measurement_count > 5))
            // {
                plane_points.push_back(c_pt);
            // }
        }

        test_pcd += plane_pcd;

        MatrixXd pts_mat(plane_points.size(), 4);
        vector<geometry_msgs::Point> vertices;

        Vector3d normal = normals_map[plane_id];

        double d = 0.0;
        for (int i = 0; i < (int)plane_points.size(); i++)
        {
            Vector3d c_pt = plane_points[i];

            geometry_msgs::Point32 p;
            p.x = (double)c_pt[0];
            p.y = (double)c_pt[1];
            p.z = (double)c_pt[2];

            Vector3d pt_ = c_pt;
            // pt_[2] = 1.0;

            d += -normal.dot(c_pt);

            pts_mat.row(i) = pt_.homogeneous().transpose();
        }
        
        d /= plane_points.size();

        Vector4d params;
        params << normal[0], normal[1], normal[2], d;

        params = fit_vertical_plane_ransac(plane_points, plane_id);
        normal = params.head<3>();

        if ((normal.norm() > 0.001) && (fabs(params[3]) > 0.001))
        {
            Vector4d normed_params(normal[0], normal[1], normal[2], params[3]);

            if (fabs(normal.dot(vertical)) < 0.5)
            {             
                if (fit_cuboid_to_point_cloud(normed_params, plane_points, vertices, vp_normals))
                {
                    create_cuboid_frame(vertices, line_list, (Ti * Tic));
                    //write_estimated_normal_error(normal , vp_normals /*ground truth*/);

                    for (int vid = 0; vid < vertices.size(); vid++)
                    {
                        frame_cloud.points.push_back(pointToPoint32(vertices[vid]));
                        plane_id_ch.values.push_back(plane_id);
                    }
                }
                else {
                    ROS_INFO("Not plotting cuboid %d with params %f %f %f %f", plane_id, normed_params[0], normed_params[1], normed_params[2], normed_params[3]); 
                }
            }
        }
        
        plane_ids.push_back(plane_id);
    }

    marker_pub.publish(line_list);

    pcl::toROSMsg(test_pcd, test_cloud);
    test_cloud.header = features_msg->header;
    frame_pub2.publish(test_cloud);

    ROS_INFO("Publising marked image");
    std_msgs::Header img_header;
    img_header = img_msg->header;
    sensor_msgs::ImagePtr marked_image_msg = cv_bridge::CvImage(img_header, sensor_msgs::image_encodings::BGR8, mask_img).toImageMsg();
    
    // Publish raw images with marked quads
    masked_im_pub.publish(marked_image_msg);

    // Process a particular plane id
    frame_cloud.channels.push_back(plane_id_ch);
    frame_pub.publish(frame_cloud);
    // sensor_msgs::PointCloud2 frame_cloud2;
    // sensor_msgs::convertPointCloudToPointCloud2(frame_cloud, frame_cloud2);
    // frame_pub2.publish(frame_cloud2);
    // ma_pub.publish(ma);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_mapper");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    load_color_palette(COLOR_PALETTE_PATH);

    // Register all subscribers
    message_filters::Subscriber<sensor_msgs::PointCloud> sub_point_cloud(n, "/point_cloud", 100);
    message_filters::Subscriber<nav_msgs::Odometry> sub_odometry(n, "/odometry", 100);
    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, "/image", 10);
    message_filters::Subscriber<sensor_msgs::Image> sub_mask(n, "/mask", 10);
    //message_filters::Subscriber<sensor_msgs::Image> sub_seg(n, "/airsim_node/drone/0/Segmentation", 10);
    //message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, "/airsim_node/drone/0/DepthPerspective", 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    //typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry> MySyncPolicyDepth;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_point_cloud, sub_odometry, sub_image, sub_mask);
    //message_filters::Synchronizer<MySyncPolicyDepth> sync_depth(MySyncPolicyDepth(2), sub_depth, sub_seg, sub_odometry);
    // message_filters::TimeSynchronizer<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image> sync(
    //     sub_point_cloud,
    //     sub_odometry,
    //     sub_image,
    //     sub_mask,
    //     2000
    // );
    sync.registerCallback(boost::bind(&mapping_callback, _1, _2, _3, _4));
    //sync_depth.registerCallback(boost::bind(&depth_image_callback, _1, _2, _3));

    // Register all publishers
    // Publish coloured point cloud
    // Publish 3D plane segments (line list or marker array)

    frame_pub = n.advertise<sensor_msgs::PointCloud>("frame_cloud", 100);
    cent_pub = n.advertise<sensor_msgs::PointCloud>("centroid_cloud", 100);
    lgoal_pub = n.advertise<sensor_msgs::PointCloud>("local_goal", 100);
    frame_pub2 = n.advertise<sensor_msgs::PointCloud2>("frame_cloud2", 100);
    masked_im_pub = n.advertise<sensor_msgs::Image>("masked_image", 10);
    marker_pub = n.advertise<visualization_msgs::Marker>("cuboids", 10);
    // ma_pub = n.advertise<visualization_msgs::MarkerArray>("centroid_segs", 100);
    //dense_pub = n.advertise<sensor_msgs::PointCloud2>("dense_cloud", 5);
    ros::spin();
}
