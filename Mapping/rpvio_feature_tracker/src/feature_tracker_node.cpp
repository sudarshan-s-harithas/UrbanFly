#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "feature_tracker.h"
#include "vp_utils.h"
#include "color_ids.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match, pub_mask_cloud;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

void processMaskMessage(const sensor_msgs::ImageConstPtr &input_mask_msg, cv::Mat &output_mask)
{
    sensor_msgs::Image output_mask_msg;

    cv_bridge::CvImagePtr input_mask_ptr;
    input_mask_ptr = cv_bridge::toCvCopy(input_mask_msg, sensor_msgs::image_encodings::BGR8);
    
    // Extract RGB mask from input mask message
    cv::Mat input_mask;
    input_mask = input_mask_ptr->image;
    
    // cv::imshow("input mask", input_mask);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // Convert RGB mask to CV_32U (32-bit single channel image)
    for (int i = 0; i < input_mask.rows; i++) {
        for (int j = 0; j < input_mask.cols; j++) {
            cv::Vec3b colors = input_mask.at<cv::Vec3b>(i, j);
            
            output_mask.at<uchar>(i, j) = (uchar)color2id(colors[0], colors[1], colors[2]);
        }
    }
        
    // cv::imshow("output mask", output_mask);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    std::cout << "I received an image at " << std::to_string(img_msg->header.stamp.toSec()) << std::endl;
}

void msk_callback(const sensor_msgs::ImageConstPtr &msk_msg)
{
    std::cout << "I received a mask at " << std::to_string(msk_msg->header.stamp.toSec()) << std::endl;
}

void callback(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::ImageConstPtr &mask_msg)
{
    // std::cout << "I received an image and a mask ! " << std::endl;
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();
        last_image_time = img_msg->header.stamp.toSec();
        return;
    }
    // detect unstable camera stream
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = img_msg->header.stamp.toSec();
    // frequency control
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr, ptr2;
    // cv_bridge::CvImagePtr mask_ptr, mask_ptr_bgr8;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    ptr2 = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);

    // const sensor_msgs::Image processed_mask = processMaskMessage(mask_msg); 
    cv::Mat pmask_img(ROW, COL, CV_8UC1, cv::Scalar(0));
    processMaskMessage(mask_msg, pmask_img);
    const cv::Mat cmask_img(pmask_img);
    
    // mask_ptr_bgr8 = cv_bridge::toCvCopy(mask_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat show_img = ptr->image;
    cv::Mat show_img2 = ptr2->image;

    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK)
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), cmask_img.rowRange(ROW * i,ROW * (i + 1)), img_msg->header.stamp.toSec());
        else
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 plane_id_of_point; 
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &plane_ids = trackerData[i].plane_ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    plane_id_of_point.values.push_back(plane_ids[j]);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(plane_id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;
            // cv::Mat mask_img;
            std::vector<int> pids = trackerData[0].plane_ids;

            std::set<int> unique_pids(pids.begin(), pids.end());
            std::map<int, int> plane_counts;
            for(auto &pid : unique_pids)
                plane_counts[pid] = 0;

            for(auto &pid : pids)
                plane_counts[pid]++;

            int largest_count = 0, largest_pid;
            for(auto &plane_count : plane_counts)
            {   
                if(plane_count.second > largest_count)
                {
                    largest_count = plane_count.second;
                    largest_pid = plane_count.first;
                }
            }
    
            sensor_msgs::PointCloud mask_cloud;
            mask_cloud.header = img_msg->header;
            mask_cloud.header.frame_id = "world";
            sensor_msgs::ChannelFloat32 mask_ids;
            sensor_msgs::ChannelFloat32 colors;
            colors.name = "rgb";

            vector<cv::Scalar> label_colors;
            label_colors.push_back(cv::Scalar(255,0,0));
            label_colors.push_back(cv::Scalar(0,255,0));
            label_colors.push_back(cv::Scalar(0,0,255));
            label_colors.push_back(cv::Scalar(255,255,0));
            label_colors.push_back(cv::Scalar(0,255,255));
            label_colors.push_back(cv::Scalar(255,0,255));

            // map<int, cv::Scalar> label_colors;
            // label_colors[39] = cv::Scalar(255,0,0);
            // label_colors[66] = cv::Scalar(255,0,255);
            // label_colors[91] = cv::Scalar(0,255,0);
            // label_colors[130] = cv::Scalar(0,0,255);
            // label_colors[162] = cv::Scalar(255,255,0);
            // label_colors[175] = cv::Scalar(0,255,255);

            vector<int> unique_pids_vec;
            for (auto& pid: unique_pids) {
                unique_pids_vec.push_back(pid);
            }

            ROS_INFO("-------------NUMBER OF UNIQUE MASKS ARE : %d; LARGEST : %d---------------", (int)unique_pids_vec.size(), largest_pid);

            Eigen::Matrix3d K;
            K << FOCAL_LENGTH, 0, COL/2,
                    0, FOCAL_LENGTH, ROW/2,
                    0, 0, 1;

            cv::Mat tmp_img = ptr->image.rowRange(0 * ROW, (0 + 1) * ROW);
            cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);
            // show_img2.copyTo(tmp_img);
            // cv::cvtColor(mask_ptr_bgr8->image, tmp_img, CV_GRAY2RGB);
                        std::vector<KeyLine> lines_klsd;
                        cv::Mat lines_lsd_descr; 
                        std::vector<cv::Point3d> vps(3);
                        std::vector<std::vector<int> > clusters(3);
                        std::vector<int> lines_vps;
                        double f = K(0, 0);
                        cv::Point2d pp(K(0, 2), K(1, 2));
                        int LENGTH_THRESH = 2;
                        
                        cv::Mat building_mask = pmask_img > 0;
                        ROS_INFO("Extracting line segments and vanishing points");
                        extract_lines_and_vps(
                            tmp_img, 
                            lines_klsd, lines_lsd_descr, 
                            vps, clusters,
                            lines_vps,
                            f, pp, LENGTH_THRESH, pmask_img
                        );

                        // if (lines_klsd.size() > 10) {
                        //     ROS_INFO("Detected vanishing points are : ");
                        //     for (size_t i = 0; i < vps.size(); i++)
                        //     {
                        //         ROS_INFO("%f, %f, %f", vps[i].x, vps[i].y, vps[i].z);
                        //     }
                        // }

            /*for (int ppi = 0; ppi < unique_pids_vec.size(); ppi++) {
                int mid = unique_pids_vec[ppi];
                // ROS_INFO("||||||||||||| ID = %d |||||||||||||||||||||||||", mid);
                if (
                    // (mid == 91) || 
                    // (mid == 39) ||
                    // (mid == 66) ||
                    // (mid == 130) || 
                    // (mid == 162) || 
                    // (mid == 194)
                    // (mid == 104)
                    // (mid == 86) ||
                    // (mid == 99) ||
                    // (mid == 81)
                    // (mid == 42) ||
                    // (mid == 104) ||
                    // (mid == 23)
                    // (mid == 44) // very bad recon, as it is parallel
                    // (mid == 208)
                    // (mid == 156)
                    // (mid == 110) // ground plane in minihattan
                    true
                    // mid == largest_pid
                    // false
                ) {
                std::vector<std::vector<cv::Point>> contours;
                
                cv::Mat mask_img = pmask_img;
                cv::Mat mask = mask_img == mid;
                cv::Mat mask_contour(ROW, COL, CV_8UC1, cv::Scalar(0));
                cv::Mat mask_filled(ROW, COL, CV_8UC1, cv::Scalar(0));
                mask_filled.setTo(cv::Scalar(255), mask);

                cv::findContours(mask_filled, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                cv::drawContours(mask_contour, contours, -1, cv::Scalar(255), 3);

                // cv::Mat mask_viz(ROW, COL, CV_8UC3, cv::Scalar(0,0,0));
                // mask_viz.setTo(label_colors[mid%6], mask);
                // mask_img.copyTo(mask_viz);
                // mask_img.copyTo(mask_viz);
                // mask_img.copyTo(mask_viz);
                // cv::Mat mask_viz2 = mask_ptr_bgr8->image;
                // mask_viz2.copyTo(tmp_img);

                // cv::addWeighted(tmp_img, 1.0, mask_viz, 0.3, 0.0, tmp_img);

                for (int i = 0; i < NUM_OF_CAM; i++)
                {

                    // Binary mask
                    cv::Mat binMask(ROW, COL, CV_8UC1, cv::Scalar(0));
                    // cv::Mat bgr[3];
                    // cv::split(mask_viz, bgr);
                    binMask.setTo(cv::Scalar(255), mask_contour);
                    // cv::threshold(bgr[1], binMask, 100, 255, CV_THRESH_BINARY);

                    // if (mid == largest_pid) {
                        // std::vector<KeyLine> lines_klsd;
                        // cv::Mat lines_lsd_descr; 
                        // std::vector<cv::Point3d> vps(3);
                        // std::vector<std::vector<int> > clusters(3);
                        // std::vector<int> lines_vps;
                        // double f = K(0, 0);
                        // cv::Point2d pp(K(0, 2), K(1, 2));
                        // int LENGTH_THRESH = 0;
                        
                        // ROS_INFO("Extracting line segments and vanishing points");
                        // extract_lines_and_vps(
                        //     tmp_img, 
                        //     lines_klsd, lines_lsd_descr, 
                        //     vps, clusters,
                        //     lines_vps,
                        //     f, pp, LENGTH_THRESH
                        // );

                        // ROS_INFO("Detected vanishing points are : ");
                        // for (size_t i = 0; i < vps.size(); i++)
                        // {
                        //     ROS_INFO("%f, %f, %f", vps[i].x, vps[i].y, vps[i].z);
                        // }
                    // }

                    // Compute the mask cloud
                    for (unsigned int u = 0; u < binMask.rows; u+=10) {
                        for (unsigned int v = 0; v < binMask.cols; v+=10) {
                            if (binMask.at<uint8_t>(u, v) == 255) {
                                geometry_msgs::Point32 p;
                                // Eigen::Vector2d a(v, u);
                                Eigen::Vector3d a(v, u, 1);
                                // Eigen::Vector3d a_;
                                Eigen::Vector3d a_;
                                // trackerData[i].m_camera->liftProjective(a, a_);
                                
                                a_ = K.inverse() * a;
                            
                                p.x = a_.x();
                                p.y = a_.y();
                                p.z = a_.z();

                                mask_cloud.points.push_back(p);
                                
                                // int rgb = 0xaaff00; float float_rgb = *reinterpret_cast<float*>(&rgb);
                                // Eigen::Vector3i pix = tmp_img.at<Eigen::Vector3i>(u, v);
                                
                                // uint8_t r = (uint8_t)label_colors[mid][0];// pix(2);
                                // uint8_t g = (uint8_t)label_colors[mid][1];// pix(1);
                                // uint8_t b = (uint8_t)label_colors[mid][2];// pix(0);
                                unsigned int b = show_img.at<uint8_t>(u, v);

                                int rgb = ((b & 0xff) << 16) + ((b & 0xff) << 8) + (b & 0xff);
                                colors.values.push_back(rgb);
                                mask_ids.values.push_back(mid);
                            }
                        }
                    }

                    // for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                    // {
                    //     double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    //     cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //     //draw speed line
                    //     /*
                    //     Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    //     Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    //     Vector3d tmp_prev_un_pts;
                    //     tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    //     tmp_prev_un_pts.z() = 1;
                    //     Vector2d tmp_prev_uv;
                    //     trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    //     cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    //     
                    //     //char name[10];
                    //     //sprintf(name, "%d", trackerData[i].plane_ids[j]);
                    //     //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                    // }
                }
                }
            }*/

            // mask_cloud.channels.push_back(colors);
            // mask_cloud.channels.push_back(mask_ids);
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            // cv::imwrite("/home/tvvsstas/rpvio_ws/src/rp-vio/data/frames_vps/c1_vp_"+to_string(pub_count) + ".png", ptr->image);
            pub_match.publish(ptr->toImageMsg());
            // pub_mask_cloud.publish(mask_cloud);
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rpvio_feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    load_color_palette(COLOR_PALETTE_PATH);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);
    // ros::Subscriber sub_msk = n.subscribe(MASK_TOPIC, 100, msk_callback);
    // message_filters::Subscriber<sensor_msgs::Image> image_sub(n, IMAGE_TOPIC, 1000);
    // message_filters::Subscriber<sensor_msgs::Image> mask_image_sub(n, MASK_TOPIC, 1000);

    // // std::cout << "Subscribed to image and mask topics : " << IMAGE_TOPIC << " and " << MASK_TOPIC << std::endl;

    // message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_sub, mask_image_sub, 1000);
    // sync.registerCallback(boost::bind(&callback, _1, _2));
    message_filters::Subscriber<sensor_msgs::Image> image1_sub(n, IMAGE_TOPIC, 10);
    message_filters::Subscriber<sensor_msgs::Image> image2_sub(n, MASK_TOPIC, 10);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), image1_sub, image2_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2));
    

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    pub_mask_cloud = n.advertise<sensor_msgs::PointCloud>("mask_cloud",1);
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */

    ros::spin();
    return 0;
}

