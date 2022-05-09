#include "local_map.h"

// For thread sync
struct SubMessages
{
    sensor_msgs::PointCloudConstPtr features_msg;
    nav_msgs::OdometryConstPtr odometry_msg;
    sensor_msgs::ImageConstPtr mask_msg;
};
queue<SubMessages> sm_queue;
condition_variable sm_cond;
mutex sm_mutex;

// Publishers
ros::Publisher clusters_pub;
ros::Publisher cuboids_pub;
ros::Publisher vertices_pub;

LocalMap previous_map;
int local_map_id = 2;
int plane_counter = 1000;

sensor_msgs::PointField getFieldWithName(string name)
{
    sensor_msgs::PointField pf;
    pf.name = name;

    return pf;
}

void process_messages()
{
    while(true)
    {
        unique_lock<mutex> lck{sm_mutex};
        sm_cond.wait(lck, []{ return !sm_queue.empty();});

        SubMessages sub_msg;

        sub_msg = sm_queue.front();
        sm_queue.pop();
        lck.unlock();

        sensor_msgs::PointCloudConstPtr features_msg = sub_msg.features_msg;
        nav_msgs::OdometryConstPtr odometry_msg = sub_msg.odometry_msg;
        sensor_msgs::ImageConstPtr mask_msg = sub_msg.mask_msg;

        LocalMap lm(features_msg, odometry_msg, mask_msg);
        lm.id = 2;//local_map_id++;
        lm.plane_counter = plane_counter;
        //lm.mPlanes = previous_map.mPlanes;
        lm.mPlaneFeatures = previous_map.mPlaneFeatures;
        lm.cluster_points();
        lm.fit_cuboids();
        lm.merge_old_map(previous_map);
        lm.fit_cuboids();
        lm.publish_clusters(clusters_pub);
        lm.publish_cuboids(cuboids_pub, vertices_pub);

        previous_map = lm;
        plane_counter = lm.plane_counter;
    }
}

void mapping_callback(
    const sensor_msgs::PointCloudConstPtr &features_msg,
    const nav_msgs::OdometryConstPtr &odometry_msg,
    // const sensor_msgs::ImageConstPtr &img_msg,
    const sensor_msgs::ImageConstPtr &mask_msg
)
{
    SubMessages sub_msgs;
    sub_msgs.features_msg = features_msg;
    sub_msgs.odometry_msg = odometry_msg;
    // sub_msgs.img_msg = img_msg;
    sub_msgs.mask_msg = mask_msg;

    unique_lock<mutex> lck{sm_mutex};
    // sm_mutex.lock();
    sm_queue.push(sub_msgs);
    // sm_mutex.unlock();
    lck.unlock();
    sm_cond.notify_one();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_mapper");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    // Register all subscribers
    message_filters::Subscriber<sensor_msgs::PointCloud> sub_point_cloud(n, MAPPER_POINT_CLOUD_TOPIC, 5);
    message_filters::Subscriber<nav_msgs::Odometry> sub_odometry(n, MAPPER_ODOMETRY_TOPIC, 5);
    message_filters::Subscriber<sensor_msgs::Image> sub_mask(n, MAPPER_MASK_TOPIC, 5);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image> MySyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(5), sub_point_cloud, sub_odometry, sub_mask);

    // message_filters::TimeSynchronizer<sensor_msgs::PointCloud, nav_msgs::Odometry, sensor_msgs::Image, sensor_msgs::Image> sync(
    //     sub_point_cloud,
    //     sub_odometry,
    //     sub_image,
    //     sub_mask,
    //     2000
    // );
    sync.registerCallback(boost::bind(&mapping_callback, _1, _2, _3));

    // Register all publishers
    // Publish coloured point cloud
    // Publish 3D plane segments (line list or marker array)

    clusters_pub = n.advertise<sensor_msgs::PointCloud2>("clusters", 5);
    cuboids_pub = n.advertise<visualization_msgs::Marker>("cuboids", 5);
    vertices_pub = n.advertise<sensor_msgs::PointCloud>("vertices", 5);
    
    thread mapping_thread{process_messages};
    ros::spin();    
}
