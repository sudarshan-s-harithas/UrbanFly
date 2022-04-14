#include <chrono>
#include "utilsPlannner.h"
#include "objects.h"
#include"parameters.h"
#include"kinodynamic_astar2.h"
#include "planner.h"
#include"STOMP.h"
// #include"CEM.h"
#include <iostream>
#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <string>
#include <thread>
#include "planner.h"
#include "CEM_Polynomial.h"

std::string ip ; 

fast_planner::KinodynamicAstar kAstar;
Initilizer::STOMP STOMPTrajectories ; 
Eigen::Vector3d goal;
ros::Publisher Centervis; 
 
// Optimization::TrajectoryOptimization CEMOptim; 

Eigen::Vector3d startVel;
Eigen::Vector3d startAcc; 
Eigen::Vector3d goalVel; 

Eigen::MatrixXd x_samples; 
Eigen::MatrixXd y_samples;
Eigen::MatrixXd z_samples; 

ros::Publisher PubSTOMP; 
ros::Publisher CEMOptimTraj;

nav_msgs::Path CEMOptimTrajectory ; 
ros::Publisher CEM_MeanTrajectory ; 

bool status ;
double deltaT = 0.1;  
nav_msgs::Path AstarTrajectory;
int numIters = 3;

ros::Publisher AstarTraj ; 
ros::Publisher CEMOptimPath;

bool In= true ; 
bool StartOver = false;

Eigen::Vector3d GoalState;
Eigen::Vector3d CurState; 
Eigen::Vector3d PrevState ;

int cnt =0 ;
int NumTraj_perturb = 100; // 500
int NumPts_perTraj ; 
bool Remap = false ;

PolynomialFormulation::CEMPolynomialFormulation CEM_polynomial; 
Bernstein::BernsteinPath  BernsteinTraj(10);
using namespace std::chrono ;

Eigen::Vector3d goalPose;
bool goalReceived = false;
bool GoalUpdate = false;

double CurrZ =0 ;  

bool ClickPointUpdate = false; 

void goal_pose_cb(const geometry_msgs::PoseStamped pose)
{
    goalReceived = true;
    double dummy=0;

    goalPose(0) =  pose.pose.position.x;//  17.6733, 5.064 , 4.0



  
    goalPose(1) = pose.pose.position.y;

    goalPose(2) = CurrZ;
    
    // std::cout<<"Enter the height at the goal point ";
    // std::cin>>goalPose(2);

    std::cout<<"Goal Pose is ...*********** "<<goalPose.transpose()<<std::endl;
    std::cout<<"\n";
    GoalUpdate = true ;
    StartOver = true;

     // std::cout<<"Enter the height at the goal point ";
     // std::cin>>dummy;
}


void UseClickedPoint( const geometry_msgs::PointStamped pose ){

    ClickPointUpdate = true; 


}

void VisulizeCenters( std::vector<Eigen::Vector3d> Centers )
{

    int numObs = Centers.size();
    int color_counter =0 ; 

    Eigen::Vector3d Center; 
    geometry_msgs::Point pt;
    visualization_msgs::Marker marker;


    for( int i =0 ; i < numObs ; i++){

        color_counter++;
        
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time();
        marker.ns = "my_namespace";
        marker.id = color_counter;

        Center = Centers.at(i);

        // std::cout << " Centers " <<  Center << " " << i << std::endl ;

        pt.x = Center.x();
        pt.y  = Center.y();
        pt.z = Center.z() ;

        marker.points.push_back(pt);


        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 1; 
        marker.color.r = 1.0 ;
        marker.color.g = 0.5;
        marker.color.b = 0.2*(numObs - float(color_counter))/numObs;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

    }

    if( numObs %2 == 1){


        pt.x = 0;
        pt.y  = 0;
        pt.z = 0 ;

        marker.points.push_back(pt);


        marker.scale.x = 0.1;
        marker.scale.y = 0.1;
        marker.scale.z = 0.1;
        marker.color.a = 1; 
        marker.color.r = 1.0 ; //(num_samples - float(color_counter))/num_samples;
        marker.color.g = 0.5;
        marker.color.b = 0.2*(numObs - float(color_counter))/numObs;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;


    }

    Centervis.publish(marker);

}


void GetSmoothnessCost(     Eigen::MatrixXd X , Eigen::MatrixXd Y ,   Eigen::MatrixXd Z  )
{

    int numPts = X.cols();

    Eigen::Vector3d Pt; 
    Eigen::Vector3d PrevPt ;
    double dt = 0.6; 
    double dist ; 
    double cost = 0 ;


    for( int i =0 ; i < numPts ; i++ ){
        if( i == 0){
            PrevPt << X( 0 , i ) , Y(0 , i ) , Z(0 , i) ; 
            continue; 
        }
        Pt <<  X( 0 , i ) , Y(0 , i ) , Z(0 , i) ; 

        dist = ( Pt - PrevPt ).norm() ;
        cost += (dist*dist)/ (dt*dt*dt*dt*dt);
        PrevPt = Pt ;
    }

    std::cout << "  Smoothness Cost =  "  <<  cost << std::endl ;


}




void current_state_callback2( const sensor_msgs::PointCloudConstPtr &frames_msg, const nav_msgs::OdometryConstPtr &odometry_msg )
{

    CurState << odometry_msg->pose.pose.position.x, odometry_msg->pose.pose.position.y,
                 odometry_msg->pose.pose.position.z ; 

    // CurState.z() = 0 ; 
    // GoalState.z() = 0;

    CurrZ =   odometry_msg->pose.pose.position.z ;




    double distTravelled = (CurState - PrevState ).norm() ;

    PrevState = CurState ;


    double dist_since_last = (GoalState - CurState).norm();   // distance between current pose and local goal 

    if( goalReceived ){

    
    if(  ((cnt) == 0  ) || ( dist_since_last < 2.5) || GoalUpdate || true){
        // GoalUpdate = false;



        std::cout << distTravelled <<  "  " << " Replanning " <<  std::endl ;
        std::cout << dist_since_last << " " << "dist_since_last" << std::endl;

    
    In = true ; 

	// Eigen::Vector3d goal( 16.0, -4 , 1 );
    goal = goalPose;

    double DistToGoal = (CurState - goal).norm() ; 
    bool Done = false ;

    if( DistToGoal < 3.0 ){

        std::cout << "*********** REACHED GOAL congratulations *******************" << std::endl ;
        Done = true ;
    }

    if( !Done){

    Eigen::Isometry3d Tic;

    Eigen::Matrix3d R; 
    Eigen::Vector3d T ; 

    R << 0,0,1,1,0,0,0,1,0 ;
    T << 0.50, 0, 0 ; 


    Tic.linear() = R  ;  //RIC[0];
    Tic.translation() = T  ; //TIC[0];



    Eigen::Vector3d trans;
    trans <<
        odometry_msg->pose.pose.position.x,
        odometry_msg->pose.pose.position.y,
        odometry_msg->pose.pose.position.z;

    double quat_x = odometry_msg->pose.pose.orientation.x;
    double quat_y = odometry_msg->pose.pose.orientation.y;
    double quat_z = odometry_msg->pose.pose.orientation.z;
    double quat_w = odometry_msg->pose.pose.orientation.w;
    Eigen::Quaterniond quat(quat_w, quat_x, quat_y, quat_z);

    Eigen::Isometry3d Ti;
    Ti.linear() = quat.normalized().toRotationMatrix();
    Ti.translation() = trans;

    // Transform to local frame
    Eigen::Vector3d local_goal = Tic.inverse() * (Ti.inverse() * goal);
    local_goal[1] = 0.0;
    double local_goal_distance = local_goal.norm();
    Eigen::Vector3d local_goal_dir = local_goal.normalized();

    Eigen::Vector3d MapStart  ; 
    MapStart << -100 , -100 , -100 ; 
    Eigen::Vector3d MapEnd ; 
    MapEnd << 100 , 100 , 100 ; 

    Eigen::Vector3d StartPose; 

    StartPose = trans; 


    std::vector<Eigen::Vector3d>  Centers ;
    std::vector<double> radius_vector; 
    double radius; 
    Eigen::Vector3d Ground;

    double GroundLocation ;
    double ground_dist = 100 ;  
    std::vector<CuboidObject> cuboids;

    for (unsigned int i = 0; i < frames_msg->points.size(); i += 8)
    {
        int p_id = (frames_msg->channels[0].values[i]);

        Point center(0, 0, 0);
        std::vector<Eigen::Vector3d> vertices;
        Eigen::Vector3d refVertex; 
        
        

        for (int vid = 0; vid < 8; vid++)
        {
            geometry_msgs::Point32 gpt =  frames_msg->points[i + vid];
            Point pt(gpt.x, gpt.y, gpt.z);
            vertices.push_back( (Ti*Tic)*pt);
            refVertex << gpt.x, gpt.y, gpt.z ;

            center += pt;

            if(gpt.y < ground_dist ){
                ground_dist = gpt.y ;
            }

        }

        Ground << 0 , ground_dist , 0; 

        center = center/8;
        center = (Ti*Tic)*center ; 

        Ground = (Ti*Tic)*Ground;  // Ground distance in world frame 


        refVertex.z() = 0 ;
        center.z() =0 ;
        radius = (center - refVertex).norm(); 


        Centers.push_back(center);
        radius_vector.push_back(radius);

        GroundLocation = Ground.z() ;

        CuboidObject cuboid = CuboidObject(center, vertices, Color::Gray(), p_id);

        cuboids.push_back(cuboid);

    }



    // VisulizeCenters( Centers );


    

    if( cnt == 0  || GoalUpdate || StartOver ){
        kAstar.init( MapStart  , MapEnd , StartPose );

        if( GoalUpdate){
            StartPose = GoalState;
        }


        GoalUpdate = false;
        StartOver =false;

        if( cnt !=0 ){
        StartPose = GoalState;
    }

    if ( cnt == 0){
        StartPose = CurState;

    }


    status  = kAstar.search( StartPose ,startVel ,  startAcc , goal  , goalVel , true /*true original */, false , 0.0 ,  Centers, radius_vector 
        , GroundLocation , cuboids );
    }
    else{
        StartPose = GoalState ;

    status  = kAstar.search( StartPose ,startVel ,  startAcc , goal  , goalVel , false, false , 0.0 ,  Centers, radius_vector, 
    GroundLocation  , cuboids);

    }

    std::vector<Eigen::Vector3d> currTraj = kAstar.getKinoTraj(deltaT);


    std::cout << currTraj.size() << "--------------" << std::endl;

    if( currTraj.size()< 2){
        StartOver =true;
        kAstar.reset(); 
    }

    if( currTraj.size() > 2){

    int numPts = currTraj.size() ; 
    // std::cout << numPts << std::endl;
    x_samples = Eigen::MatrixXd::Zero( 1 ,numPts); 
    y_samples = Eigen::MatrixXd::Zero(1 ,numPts);
    z_samples= Eigen::MatrixXd::Zero(1 ,numPts ); 
    int pos_cnt =0 ;

    for(auto i = currTraj.begin(); i!=currTraj.end(); i++)
    {
                geometry_msgs::PoseStamped p;
                Eigen::Vector3d pos = *i;


                p.pose.position.x = float( pos(0) );
                p.pose.position.y = float(pos(1) );
                p.pose.position.z = float(pos(2) );


                x_samples(0 , pos_cnt) = pos(0);
                y_samples(0, pos_cnt ) =pos(1);
                z_samples(0 , pos_cnt ) = pos(2);
                pos_cnt +=1; 


                p.pose.orientation.w = float(1.0);

                AstarTrajectory.poses.push_back(p);
                AstarTrajectory.header.stamp = ros::Time::now();
                AstarTrajectory.header.frame_id = "/world";
    }

    AstarTraj.publish(AstarTrajectory);
    // PrevState = CurState ;


    std::vector<Eigen::MatrixXd> initTrajectory;


    initTrajectory.push_back(x_samples );
    initTrajectory.push_back(y_samples);
    initTrajectory.push_back(z_samples);

    NumPts_perTraj = numPts ; 
    bool PubSTOMP_bin = true ;

    std::vector<Eigen::MatrixXd> STOMPTraj ;

    STOMPTraj = STOMPTrajectories.PerturbAstar( initTrajectory , NumTraj_perturb ,
    NumPts_perTraj ,PubSTOMP , PubSTOMP_bin  );

    Eigen::MatrixXd xPts;
    Eigen::MatrixXd yPts;
    Eigen::MatrixXd zPts; 
        std::cout << "Here " << std::endl;


    xPts = STOMPTraj.at(0);
    yPts = STOMPTraj.at(1);
    zPts = STOMPTraj.at(2);

    std::cout << "Here " << std::endl;

    std::vector<Eigen::MatrixXd> CEMOptimizedTraj; 
    // auto start = high_resolution_clock::now();

    // CEMOptimizedTraj = CEMOptim.CrossEntropyOptimize(xPts ,yPts , zPts , numIters ,   Centers ,CEMOptimTraj , PubSTOMP , cuboids  , CEM_MeanTrajectory);
    auto start = high_resolution_clock::now();

   CEMOptimizedTraj = CEM_polynomial.CrossEntropyOptimize_Polynomial( BernsteinTraj , x_samples ,y_samples , z_samples , numIters ,Centers , CEMOptimTraj , PubSTOMP , cuboids , CEM_MeanTrajectory );
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout<< duration.count() << " COmputation Time " << std::endl;


    Eigen::MatrixXd X;
    Eigen::MatrixXd Y ; 
    Eigen::MatrixXd Z; 

    X = CEMOptimizedTraj.at(0);
    Y = CEMOptimizedTraj.at(1);
    Z = CEMOptimizedTraj.at(2);

    GetSmoothnessCost( X, Y , Z);

    int num = X.cols(); 

    std::cout << num << " Number  of Points "  << X.rows() << std::endl ;

    for(int i =0 ; i< num ; i++)
    {

        geometry_msgs::PoseStamped p2;

        p2.pose.position.x = X(0 , i );
        p2.pose.position.y= Y(0 , i);
        p2.pose.position.z = Z( 0, i);

        p2.pose.orientation.w = 1.0 ; 

        CEMOptimTrajectory.poses.push_back(p2);
        CEMOptimTrajectory.header.stamp = ros::Time::now();

        CEMOptimTrajectory.header.frame_id = "world";
    }

    // std::this_thread::sleep_for(std::chrono::seconds(5));

    CEMOptimPath.publish(CEMOptimTrajectory);



    std::cout << " Start " << " " << x_samples(0, 0 ) << " " << y_samples(0 ,0 ) << " " << z_samples(0,0) << std::endl; 
    std::cout << " End " << "  " << x_samples(0 ,numPts-1) << " " << y_samples(0 ,numPts-1 ) << " " << z_samples(0,numPts-1) << std::endl; 
    std::cout << " ---------------------------" << std::endl;

    GoalState <<  x_samples(0 ,numPts-1) , y_samples(0 ,numPts-1 ) , z_samples(0,numPts-1) ;


    kAstar.reset(); 
}

}


}

std::cout << cnt << std::endl ;
cnt += 1; 

}



// AstarTraj.publish(AstarTrajectory);



}



int main(int argc, char **argv)
{

	ros::init(argc, argv, "VinsPlanner");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    message_filters::Subscriber<sensor_msgs::PointCloud> sub_frame_cloud(n, "/rpvio_mapper/frame_cloud", 20); // /rpvio_mapper/vertices /rpvio_mapper/frame_cloud
    message_filters::Subscriber<nav_msgs::Odometry> sub_odometry(n, "/vins_estimator/odometry", 20);
    PubSTOMP = n.advertise<visualization_msgs::Marker>( "/STOMP_vis", 0 );
    CEMOptimTraj = n.advertise<visualization_msgs::Marker>( "/OptimizedTraj", 0 );
    CEMOptimPath = n.advertise<nav_msgs::Path>("/CEMPath" , 0 ); 
    CEM_MeanTrajectory = n.advertise<nav_msgs::Path>("/CEM_mean_Path" , 0 );
    ros::Subscriber goal_sub         = n.subscribe<geometry_msgs::PoseStamped>("/move_base_simple/goal",1,goal_pose_cb);
    ros::Subscriber clickedSub         = n.subscribe<geometry_msgs::PointStamped>("/clicked_point",1, UseClickedPoint );

    Centervis = n.advertise<visualization_msgs::Marker>("/CenterObs" , 0 );

    startVel = Eigen::Vector3d::Zero();
    startAcc = Eigen::Vector3d::Ones(); 
    goalVel  = Eigen::Vector3d::Zero();

    PrevState = Eigen::Vector3d::Zero();

    AstarTraj = n.advertise<nav_msgs::Path>( "/AstarTraj", 0 );
    kAstar.setParam(n);

    GoalState = Eigen::Vector3d::Zero(); 





    message_filters::TimeSynchronizer<sensor_msgs::PointCloud, nav_msgs::Odometry> sync2(
        sub_frame_cloud,
        sub_odometry, 
        100);
    
  
    sync2.registerCallback(boost::bind(&current_state_callback2, _1, _2));

    // ros::Timer timer = nh.createTimer(ros::Duration(1), timerCallback);


	

   

    ros::spin();







	return 0; 

}