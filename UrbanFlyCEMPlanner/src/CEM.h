#include <random>
#include<algorithm>
#include"MMD.h"
#include <math.h>
#include <bits/stdc++.h>
#include <iostream>
#include <fstream>


#include <chrono>
using namespace std::chrono;

MMDFunctions::MMD_variants MMDF;



namespace Optimization
{


	class TrajectoryOptimization
	{

	public:

		int CEM( Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , Eigen::MatrixXd PlaneParams ,
		ros::Publisher OptimTraj , ros::Publisher SampleTraj );

		std::vector<Eigen::MatrixXd> CrossEntropyOptimize(  Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , std::vector<Eigen::Vector3d>  Centers,  
		ros::Publisher OptimTraj , ros::Publisher SampleTraj , std::vector<CuboidObject> cuboids , ros::Publisher  CEM_MeanTrajectory  );
		double MMDwithNormalUncertainity( std::vector<CuboidObject> cuboids , Eigen::Vector3d Qpt , int numSamples_normal_uncertainity , bool Ismean );

		void SetUncertainityMeasurments();


	protected:



		int numDegreeSamples  = 100 ; 
		int  numDistanceSamples  = 100;
		Eigen::MatrixXd NormalDistribution ; 
		Eigen::MatrixXd DistanceDistribution ;
		int iter_cnt = 0 ; 


	};
} 




Eigen::MatrixXd MeasureDistance( Eigen::MatrixXd PlaneParams , Eigen::Vector3d QPt   )
{

Eigen::Vector3d normal;
double D1, D2; 
int numNormals=  PlaneParams.rows();
Eigen::MatrixXd Dist( numNormals , 1); 


for (int i=0; i<numNormals ; i++ ){



	normal <<  PlaneParams.row(i)(0) ,  PlaneParams.row(i)(1) , PlaneParams.row(i)(2)  ;
	D2 = normal.norm();

	D1 = normal.dot(QPt) + PlaneParams.row(i)(3);
	double temp = D1/D2 ; 

	// if( temp < 0 ){
	// 	temp = 100; 
	// }

	Dist(i , 0 ) = temp;

	// std::cout<< D1/D2 <<std::endl;

}

return Dist;

}


/*
double MMDCost( double  distance , int num_samples_distance   )
{
	double MMD_cost; 

	int num_samples_of_distance_distribution = num_samples_distance ; 
	std::random_device rd{};
	std::mt19937 gen{rd()};

	std::normal_distribution<> noise{0,2};    
	Eigen::MatrixXf noise_distribution(1, num_samples_of_distance_distribution);
	Eigen::MatrixXf noise_distribution2(1, num_samples_of_distance_distribution);
	Eigen::MatrixXf radius(1, num_samples_of_distance_distribution);  
     // radius.setOnes();
	radius = Eigen::MatrixXf::Constant( 1, num_samples_of_distance_distribution, 0.5); 
	Eigen::MatrixXf actual_distance( 1, num_samples_of_distance_distribution);
	Eigen::MatrixXf actual_distribution( 1, num_samples_of_distance_distribution);
	Eigen::MatrixXf zero_matrix( 1, num_samples_of_distance_distribution);

	zero_matrix.setZero();


    using normal_dist   = std::normal_distribution<>;
    using discrete_dist = std::discrete_distribution<std::size_t>;


    auto G = std::array<normal_dist, 4>{
        normal_dist{0, 0.5}, // mean, stddev of G[0]
        normal_dist{0, 1.0}, // mean, stddev of G[1]
        normal_dist{0, 1.75} , // mean, stddev of G[2]
         normal_dist{0, 1.25}  // mean, stddev of G[3]
    };

    auto w = discrete_dist{
        0.5, // weight of G[0]
        0.35, // weight of G[1]
        0.1,  // weight of G[2]
        0.05  // weight of G[2]
    };

    for (int i=0 ; i<num_samples_of_distance_distribution; i++){

    	auto index = w(gen);
    	auto temp_noise_val = G[index](gen);
    	noise_distribution(0 ,i) = radius(0 ,i) - temp_noise_val - distance ;
    	noise_distribution2(0 ,i) = temp_noise_val;
    }

    actual_distribution = zero_matrix.cwiseMax( noise_distribution );

    MMD_cost = MMDF.MMD_transformed_features(actual_distribution) ;

    return MMD_cost ;

}
*/

Eigen::Vector3d  DetermineCenter( std::vector<Eigen::Vector3d> vertices){

	Eigen::Vector3d Center; 
	Center = Eigen::Vector3d::Zero() ; 

	for(int i=0 ;i < vertices.size() ; i++){

		Center += vertices.at(i); 
	}

	return  Center / vertices.size() ;
}

double MeasureSDFCollision( Eigen::Vector3d R , Eigen::Vector3d Q   )
{

	double distance ; 

	Eigen::Vector3d Z;
	Z = Eigen::Vector3d::Zero() ;

	Eigen::Vector3d T ; 
	T =  Q.cwiseAbs() -  R ;
	distance = T.cwiseMax(Z ).norm() ; 

	return  distance ; 
}

Eigen::Vector3d TransformQpt( Eigen::Vector3d Qpt , double epsilon , double theta , Eigen::Vector3d CubeOrigin )
{

	Eigen::Matrix3d R ;

	double angle = theta + epsilon*3.14/180 ; 

	R << cos( angle ) , -sin(angle) , 0 , 
		 sin(angle) , cos(angle) , 0 , 
		 0 ,  0 , 1 ;

	Eigen::MatrixXd T ; 
	T = -R*CubeOrigin ;

	Eigen::Vector3d Q ;
	Q = R*Qpt + T;

	return Q ; 
}


void Optimization::TrajectoryOptimization::SetUncertainityMeasurments()
{

	NormalDistribution = Eigen::MatrixXd::Constant( 1, numDegreeSamples, 0); 
	DistanceDistribution = Eigen::MatrixXd::Constant(  numDistanceSamples , 3 ,  0); 
    // std::cout<< " Saved the values 0  " <<  NormalDistribution.cols()   << std::endl ;

	std::random_device rd{};
	std::mt19937 gen{rd()};

    using normal_dist   = std::normal_distribution<>;
    using discrete_dist = std::discrete_distribution<std::size_t>;

    auto G = std::array<normal_dist, 4>{
        normal_dist{-0.592234, 5.845}, // mean, stddev of G[0]
        normal_dist{49.1803, 25.243}, // mean, stddev of G[1]
        normal_dist{ -28.4073, 29.4073} , // mean, stddev of G[2]
         normal_dist{7.798, 11.472}  // mean, stddev of G[3]
    };

    auto w = discrete_dist{
        0.7198, // weight of G[0]
        0.0473, // weight of G[1]
        0.102,  // weight of G[2]
        0.1308  // weight of G[2]
    };

    // Eigen::MatrixXd NormalDistribution( 1 ,numDegreeSamples  ) ;

    for (int i=0 ; i<numDegreeSamples; i++){

    	auto index = w(gen);
    	auto temp_noise_val = G[index](gen);
    	NormalDistribution(0 ,i) =  temp_noise_val ;
    }

    using normal_dist_   = std::normal_distribution<>;
    using discrete_dist_ = std::discrete_distribution<std::size_t>;

    auto G_distance = std::array<normal_dist_, 4>{
        normal_dist_{0, 1.2}, // mean, stddev of G[0]
        normal_dist_{3.7749, 8.5}, // mean, stddev of G[1]
        normal_dist_{0,0.5 } , // mean, stddev of G[2]
         normal_dist_{-7.798, 7.8}  // mean, stddev of G[3]
    };

    auto w_distance = discrete_dist_{
        0.278, // weight of G[0]
        0.02, // weight of G[1]
        0.9,  // weight of G[2] 0.68
        0.01  // weight of G[2]
    };

    // Eigen::MatrixXd DistanceDistribution(  numDistanceSamples , 3  ) ;

    for (int i=0 ; i<numDistanceSamples; i++){

    	auto index_ = w_distance(gen);
    	auto temp_noise_val_0 = G[index_](gen);
    	auto temp_noise_val_1 = G[index_](gen);
    	auto temp_noise_val_2 = G[index_](gen);

    	DistanceDistribution( i , 0 ) =  temp_noise_val_0 ;
    	DistanceDistribution( i , 1 ) =  temp_noise_val_1 ;
    	DistanceDistribution( i , 2 ) =  temp_noise_val_2 ;
    }

    // std::cout<< " Saved the values  " << std::endl ;
    // std::cout << DistanceDistribution( 0 , 0 ) << "   SEE here **********" << std::endl ;



}

double Optimization::TrajectoryOptimization::MMDwithNormalUncertainity( std::vector<CuboidObject> cuboids , Eigen::Vector3d Qpt , int numSamples_normal_uncertainity  ,
	bool Ismean)
{
	iter_cnt += 1 ; 

	std::vector<Eigen::Vector3d> vertices_per_cuboid ; 
	double MMD_cost =0 ; 
	Eigen::Vector3d Center_per_cuboid ; 
	double Q_Center_distance ; 
	double theta = 0 ; // theta is 0 comes from the manhatten assumption it has to be updated 
	numDegreeSamples = numSamples_normal_uncertainity ;
	numDistanceSamples = numSamples_normal_uncertainity ;

	Eigen::MatrixXf actual_distribution( 1, numSamples_normal_uncertainity);
	Eigen::MatrixXf zero_matrix( 1, numSamples_normal_uncertainity);

	zero_matrix.setZero();

	Eigen::MatrixXf radius(1, numSamples_normal_uncertainity);  

	radius = Eigen::MatrixXf::Constant( 1, numSamples_normal_uncertainity,  0.75 ); 

/*
	std::random_device rd{};
	std::mt19937 gen{rd()};

    using normal_dist   = std::normal_distribution<>;
    using discrete_dist = std::discrete_distribution<std::size_t>;

    auto G = std::array<normal_dist, 4>{
        normal_dist{-0.592234, 5.845}, // mean, stddev of G[0]
        normal_dist{49.1803, 25.243}, // mean, stddev of G[1]
        normal_dist{ -28.4073, 29.4073} , // mean, stddev of G[2]
         normal_dist{7.798, 11.472}  // mean, stddev of G[3]
    };

    auto w = discrete_dist{
        0.7198, // weight of G[0]
        0.0473, // weight of G[1]
        0.102,  // weight of G[2]
        0.1308  // weight of G[2]
    };

    Eigen::MatrixXd NormalDistribution( 1 ,numDegreeSamples  ) ;

    for (int i=0 ; i<numDegreeSamples; i++){

    	auto index = w(gen);
    	auto temp_noise_val = G[index](gen);
    	NormalDistribution(0 ,i) =  temp_noise_val ;
    }

    using normal_dist_   = std::normal_distribution<>;
    using discrete_dist_ = std::discrete_distribution<std::size_t>;

    auto G_distance = std::array<normal_dist_, 4>{
        normal_dist_{0, 1.2}, // mean, stddev of G[0]
        normal_dist_{3.7749, 8.5}, // mean, stddev of G[1]
        normal_dist_{0,0.5 } , // mean, stddev of G[2]
         normal_dist_{-7.798, 7.8}  // mean, stddev of G[3]
    };

    auto w_distance = discrete_dist_{
        0.278, // weight of G[0]
        0.02, // weight of G[1]
        0.69,  // weight of G[2]
        0.01  // weight of G[2]
    };

    Eigen::MatrixXd DistanceDistribution(  numDistanceSamples , 3  ) ;

    for (int i=0 ; i<numDistanceSamples; i++){

    	auto index_ = w_distance(gen);
    	auto temp_noise_val_0 = G[index_](gen);
    	auto temp_noise_val_1 = G[index_](gen);
    	auto temp_noise_val_2 = G[index_](gen);

    	DistanceDistribution( i , 0 ) =  temp_noise_val_0 ;
    	DistanceDistribution( i , 1 ) =  temp_noise_val_1 ;
    	DistanceDistribution( i , 2 ) =  temp_noise_val_2 ;
    }

*/
    Eigen:Vector3d Qpt_transformed ;
    float distance_ ;



    Eigen::MatrixXf Distance_Measurments( 1 , numSamples_normal_uncertainity )  ;

    std::vector<Eigen::Vector3d> vertices_cuboid ;
    double val ; 

	for( int i=0 ; i < cuboids.size() ; i++){

		vertices_cuboid = cuboids[i].vertices_ ; 
		Center_per_cuboid = DetermineCenter( vertices_cuboid ) ;
		Q_Center_distance =  ( Center_per_cuboid - Qpt ).norm() ;

		if( Q_Center_distance > 3){
			continue ;
		}

		for( int i =0 ; i < numSamples_normal_uncertainity ; i++ ){

			Qpt_transformed = TransformQpt(  Qpt , NormalDistribution( 0, i) , theta , Center_per_cuboid );
			distance_ =  float ( MeasureSDFCollision( Qpt_transformed , DistanceDistribution.row(i)  ) );
			Distance_Measurments( 0 , i ) = radius( 0, i) - distance_ ;
		}

		actual_distribution = zero_matrix.cwiseMax( Distance_Measurments ) ;
		val = MMDF.MMD_transformed_features_RBF(actual_distribution) ;
		MMD_cost += val ; 

	}
/*
	if( Ismean){
	int zero_cnt =0 ;

	for( int i =0 ; i<actual_distribution.cols() ; i++  ){

		if( actual_distribution( 0, i ) != 0  ){
			zero_cnt += 1; 
		}
	}
	std::cout << zero_cnt <<   "  " <<  actual_distribution.cols() << std::endl ;
	}
*/

	return MMD_cost ;


}


Eigen::MatrixXd GetIndex( std::vector<double>  Cost , int num_top_samples)
{

	std::vector<double> costTrajsorted = Cost;
	std::sort(costTrajsorted.begin(), costTrajsorted.end());
	Eigen::MatrixXd Indices( num_top_samples , 1) ;

	for(int i=0 ; i < num_top_samples ; i++){
		double valOptim = Cost.at(i);
		auto index = std::find(Cost.begin(), Cost.end(), valOptim);
		Indices( i ,0) = double( index - Cost.begin()) ; 
	}

	return Indices;

}


std::vector<Eigen::MatrixXd> UpdateMeanTraj(  Eigen::MatrixXd  TopIndices , Eigen::MatrixXd x_samples , Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_top_samples, 
	int numPts_perTraj  )
{

	// Eigen::MatrixXd TopX(num_top_samples ,numPts_perTraj  ) ;
	// Eigen::MatrixXd TopY(num_top_samples ,numPts_perTraj  )  ;
	// Eigen::MatrixXd TopZ(num_top_samples ,numPts_perTraj  )  ;

	Eigen::MatrixXd MeanX( 1 ,numPts_perTraj );
	Eigen::MatrixXd MeanY( 1 ,numPts_perTraj );
	Eigen::MatrixXd MeanZ( 1 ,numPts_perTraj );
	MeanX = Eigen::MatrixXd::Zero(1, numPts_perTraj);
	MeanY = Eigen::MatrixXd::Zero(1, numPts_perTraj);
	MeanZ = Eigen::MatrixXd::Zero(1, numPts_perTraj);



	for(int i =0 ;i <num_top_samples ; i++){
		int trajnum = TopIndices(i);

		for( int j =0 ; j < numPts_perTraj ; j++){
			// TopX( i , j  ) = x_samples(trajnum , j );
			// TopY( i , j  ) = y_samples(trajnum , j );
			// TopZ( i , j  ) = z_samples(trajnum , j );

			MeanX( 0, j ) += x_samples(trajnum , j )/num_top_samples ;
			MeanY(0, j ) += y_samples(trajnum , j )/num_top_samples ; 
			MeanZ(0 , j ) +=  z_samples(trajnum , j )/num_top_samples; 
		}
	}


	std::vector<Eigen::MatrixXd> MeanPts; 
	MeanPts.push_back(MeanX);
	MeanPts.push_back(MeanY);
	MeanPts.push_back(MeanZ);

	return MeanPts;


}


int VisOptimTraj( Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples  , ros::Publisher OptimTraj , ros::Publisher CEM_MeanTrajectory )
{

	int numPts_perTraj = x_samples.cols();

	visualization_msgs::Marker marker;
	marker.type = visualization_msgs::Marker::LINE_LIST;
	marker.action = visualization_msgs::Marker::ADD;
	geometry_msgs::Point pt;


	for( int j =0 ; j< numPts_perTraj ; j++){

		marker.header.frame_id = "world";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = j;

		pt.x = x_samples(0, j) ;
		pt.y  = y_samples(0, j) ; 
		pt.z = z_samples(0, j);

		marker.points.push_back(pt);

		marker.scale.x = 0.1;
		marker.scale.y = 0.1;
		marker.scale.z = 0.1;
		marker.color.a = 1; 
		marker.color.r = 0.0 ; //(num_samples - float(color_counter))/num_samples;
		marker.color.g = 1.0;
		marker.color.b = 1.0 ; // *(numPts_perTraj - float(j))/numPts_perTraj;

		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;

	}
	int j = numPts_perTraj ;

	if( numPts_perTraj%2 == 1 )
	{


	marker.header.frame_id = "world";
	marker.header.stamp = ros::Time();
	marker.ns = "my_namespace";
	marker.id = j+1;

	pt.x = x_samples(0, j-1) ;
	pt.y  = y_samples(0, j-1) ; 
	pt.z = z_samples(0, j-1);

	marker.points.push_back(pt);

	marker.scale.x = 0.1;
	marker.scale.y = 0.1;
	marker.scale.z = 0.1;
	marker.color.a = 1; 
	marker.color.r = 0.0 ; //(num_samples - float(color_counter))/num_samples;
	marker.color.g = 1.0;
	marker.color.b = 1.0*(numPts_perTraj - float(j+1))/numPts_perTraj;

	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;

	}
	nav_msgs::Path CEM_mean_path ; 


	for( int i =0 ; i < numPts_perTraj ; i ++){

		geometry_msgs::PoseStamped p2;

		p2.pose.position.x = x_samples(0,  i) ;
		p2.pose.position.y = y_samples(0, i) ;
		p2.pose.position.z = z_samples( 0 , i );
		p2.pose.orientation.w = 1.0 ; 

		CEM_mean_path.poses.push_back(p2);
		CEM_mean_path.header.stamp =ros::Time::now() ;
		CEM_mean_path.header.frame_id = "world";
	}

	CEM_MeanTrajectory.publish(CEM_mean_path);


	OptimTraj.publish(marker);
	return 0; 


}



int VisulizeSampleTrajectory( Eigen::MatrixXd  x_samples , Eigen::MatrixXd y_samples , Eigen::MatrixXd z_samples , ros::Publisher SampleTraj  )
{



int color_counter = 0;
int num_samples = y_samples.rows() ; 
int num_pts_per_traj = y_samples.cols() ; 

for( int i =0 ; i < num_samples ; i++){
color_counter++;
visualization_msgs::Marker marker;
marker.type = visualization_msgs::Marker::LINE_LIST;
marker.action = visualization_msgs::Marker::ADD;

geometry_msgs::Point pt;

for( int j =0 ; j< num_pts_per_traj ; j++){

// visualization_msgs::Marker marker;
marker.header.frame_id = "world";
marker.header.stamp = ros::Time();
marker.ns = "my_namespace";
marker.id = color_counter;


pt.x = x_samples( i, j );
pt.y  = y_samples( i, j );
pt.z =z_samples( i, j );

marker.points.push_back(pt);


marker.scale.x = 0.1;
marker.scale.y = 0.1;
marker.scale.z = 0.1;
marker.color.a = 1; 
marker.color.r = 1.0 ; //(num_samples - float(color_counter))/num_samples;
marker.color.g = 0.5;
marker.color.b = 0.2*(num_samples - float(color_counter))/num_samples;
marker.pose.orientation.x = 0.0;
marker.pose.orientation.y = 0.0;
marker.pose.orientation.z = 0.0;
marker.pose.orientation.w = 1.0;
// std::cout << TopX( i, j ) << "  " << TopY( i, j ) << "  " << TopZ( i, j ) << std::endl;

}

	if( num_pts_per_traj%2 == 1 )
	{

		marker.header.frame_id = "world";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = color_counter;

		pt.x = x_samples( i, num_pts_per_traj-1 );
		pt.y  = y_samples( i, num_pts_per_traj -1);
		pt.z =z_samples( i, num_pts_per_traj -1 );

		marker.points.push_back(pt);

		marker.scale.x = 0.1;
		marker.scale.y = 0.1;
		marker.scale.z = 0.1;
		marker.color.a = 1; 
		marker.color.r = 1.0 ; //(num_samples - float(color_counter))/num_samples;
		marker.color.g = 0.5;
		marker.color.b = 0.2*(num_samples - float(color_counter))/num_samples;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;

	}

SampleTraj.publish( marker);
} 


return 0; 

}




std::vector<Eigen::MatrixXd> PerturbTraj( Eigen::MatrixXd x_samples_iter , Eigen::MatrixXd y_samples_iter, Eigen::MatrixXd z_samples_iter, 
  int num_samples , ros::Publisher SampleTraj , bool PubTraj  , int NumTrajSamples  , int iterNum  )
{

	std::default_random_engine de(time(0));


	std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;

    Eigen::Vector3d Start ;
    Start << -1,0,1 ; 
    Eigen::Vector3d Goal ; 
    Goal << 5,5,5; 




	// num_samples = number of samples in trajectory 

	double dex =  (Goal.x() - Start.x())/  num_samples;
	double dey =  (Goal.y() - Start.y())/  num_samples;
	double dez =  (Goal.z() - Start.z())/  num_samples;

	Eigen::MatrixXd StraightTraj(  num_samples+1 , 3 );

	StraightTraj.row(0) = Start ; 
	Eigen::Vector3d Temp; 
	double x,y,z; 



	for(int i =0 ; i < num_samples ; i++ ){

		x = Start.x() + dex*(i+1);
		y = Start.y() + dey*(i+1);
		z = Start.z() + dez*(i+1);

		Temp << x,y,z; 
		StraightTraj.row(i+1) = Temp; 

	}

	Eigen::MatrixXd A( num_samples , num_samples) ;
	Eigen::MatrixXd A_diff( num_samples-1 , num_samples) ;

	Eigen::MatrixXd A_diff2( num_samples-2 , num_samples) ;



	A = Eigen::MatrixXd::Identity( num_samples , num_samples );

	for(int i =0 ; i < num_samples-1 ; i++){
		for(int j =0 ; j < num_samples ; j++){

			A_diff(i, j ) =  A(i+1 , j ) - A(i, j ); 
		}
	}

	
	for(int i =0 ; i < num_samples-2 ; i++){
		for(int j =0 ; j < num_samples ; j++){

			A_diff2(i, j ) =  A_diff(i+1 , j ) - A_diff(i, j ); 
		}
	}

	Eigen::MatrixXd A_mat( num_samples+2 , num_samples);

	// std::cout<< A_diff2.rows() << "  " << A_diff2.cols() <<  std::endl;
	A_mat = Eigen::MatrixXd::Zero(  num_samples+2 , num_samples );
	A_mat(0,0) = 1;
	A_mat(1,0) = -2;
	A_mat(1,1) = 1 ;

	A_mat.block( 2, 0 , num_samples-2 , num_samples  ) = A_diff2 ; 

	A_mat( num_samples  ,num_samples-1   ) = -2;
	A_mat( num_samples  ,num_samples  -2 ) = 1;
	A_mat( num_samples +1  ,num_samples-1  ) = 1;

	A_mat = - A_mat;

	Eigen::MatrixXd R ;
	Eigen::MatrixXd cov ; 

	R = A_mat.transpose() *A_mat; 
	if( iterNum > 0 ){
	cov = (0.005/iterNum )*R.inverse(); 
	}
	else{
		cov = 0.0005*R.inverse(); 

	}

	// std::cout<< cov.rows() << "  " << cov.cols() <<  std::endl;


	Eigen::MatrixXd mu = Eigen::MatrixXd::Zero(num_samples, 1);


    Eigen::EigenMultivariateNormal<double> *normX_solver = new Eigen::EigenMultivariateNormal<double>(mu, 0.5*cov, true, dis(gen));
    Eigen::EigenMultivariateNormal<double> *normY_solver = new Eigen::EigenMultivariateNormal<double>(mu, 0.5*cov, true, dis(gen));
    Eigen::EigenMultivariateNormal<double> *normZ_solver = new Eigen::EigenMultivariateNormal<double>(mu, 0.5*cov, true, dis(gen));


    Eigen::MatrixXd eps_kx = normX_solver->samples(NumTrajSamples).transpose();
    Eigen::MatrixXd eps_ky = normY_solver->samples(NumTrajSamples).transpose();
    Eigen::MatrixXd eps_kz = normZ_solver->samples(NumTrajSamples).transpose();

    double x_fin = Goal.x();
    double y_fin = Goal.y();
    double z_fin = Goal.z();
    int t_fin =5 ;

    // std::cout << " Got till here 456 " << std::endl ;

    Eigen::VectorXd t_interp = Eigen::VectorXd::LinSpaced(num_samples, 0, t_fin);
    Eigen::VectorXd x_interp = x_samples_iter.transpose()  ; //(Start.x() + ((x_fin-Start.x())/t_fin) * t_interp.array()).matrix();
    Eigen::VectorXd y_interp = y_samples_iter.transpose() ; //(Start.y() + ((y_fin-Start.y())/t_fin) * t_interp.array()).matrix();
    Eigen::VectorXd z_interp = z_samples_iter.transpose() ;//(Start.z() + ((z_fin-Start.z())/t_fin) * t_interp.array()).matrix();

    // std::cout << " Got till here 463 " << std::endl ;

    Eigen::MatrixXd x_samples(eps_kx.rows(), eps_kx.cols());
    Eigen::MatrixXd y_samples(eps_kx.rows(), eps_kx.cols());
    Eigen::MatrixXd z_samples(eps_kx.rows(), eps_kx.cols());	

    // std::cout << " Got till here 469 " << std::endl ;

    x_samples = eps_kx;
    x_samples.rowwise() += x_interp.transpose(); //  shape = ( num_trajectories , num_samples_per_trajectory)
    y_samples = 0.0*eps_ky;
    y_samples.rowwise() += y_interp.transpose();
    z_samples = eps_kz;
    z_samples.rowwise() += z_interp.transpose();

    // std::cout << " Got till here 456 " << std::endl ;

    Eigen::Vector3d var_vector;
    var_vector << 0.2,0.2,0.2;

    /*

    for( int i =0 ; i <  num_samples ; i++ ){

    	std::normal_distribution<double> ndX( x_interp(i), var_vector.x() );
    	std::normal_distribution<double> ndY(y_interp( i ), var_vector.y());
    	std::normal_distribution<double> ndZ(z_samples(i), var_vector.z());



    	for( int j =0 ; j <  NumTrajSamples ; j++){

    		if( i == 0 || i == (num_samples -1) ){
    			x_samples( j , i ) = x_interp(i);
    			y_samples(j , i ) = y_interp(i);
    			z_samples(j, i ) =z_interp(i) ; 

    			break;
    		}

    		x_samples( j , i ) = ndX(de); 
    		y_samples(j , i ) = ndY(de);
    		z_samples(j , i ) = ndZ(de);
    	}
    }*/





    std::vector<Eigen::MatrixXd> STOMPTraj;
    STOMPTraj.push_back( x_samples);
    STOMPTraj.push_back( y_samples);
    STOMPTraj.push_back( z_samples);



    if( PubTraj == true)
    {

    	VisulizeSampleTrajectory(   x_samples ,  y_samples , z_samples ,  SampleTraj  ) ;
    }


	return STOMPTraj; 

}


double MeasureSphericalDistance( std::vector<Eigen::Vector3d> Centers , Eigen::Vector3d QPt  )
{

  // std::cout << " Enter Collision Function " << std::endl ;

  Eigen::Vector3d Center; 
  double dist ;
  std::vector<double> distances ; 
  double minDist ;

  for( auto & it: Centers ){

    Center = it ;

    dist = ( Center - QPt).norm(); 
    distances.push_back(dist); 
}


  minDist = *min_element(distances.begin(), distances.end());

  // std::cout << minDist << " CEM " <<  std::endl ;

  minDist -= 1 ; 



  return minDist ;

}


double GetPlaneCollisionDistance( std::vector<CuboidObject> cuboids , Eigen::Vector3d Qpt )
{

  std::vector<double> DistMeasurments ; 
  double dist ; 

  for( int i =0 ; i < cuboids.size() ; i++){

    dist = cuboids[i].getDistanceToPoint(Qpt); 
    DistMeasurments.push_back(dist);
  }

  double minDist;
  minDist= *min_element(DistMeasurments.begin(), DistMeasurments.end()) ;

  return minDist;

}



std::vector<Eigen::MatrixXd> Optimization::TrajectoryOptimization::CrossEntropyOptimize( Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , 
	std::vector<Eigen::Vector3d> Centers,   ros::Publisher OptimTraj , ros::Publisher SampleTraj , std::vector<CuboidObject> cuboids , ros::Publisher CEM_MeanTrajectory )
{

	// x_samples shape is ( num_trajs , num_pts_per_traj )

	std::vector<Eigen::MatrixXd> CEMOptimizedTraj; 


	std::vector<Eigen::MatrixXd> MeanTraj; 

	SetUncertainityMeasurments();



	int numTrajs = x_samples.rows();
	int num_top_samples = int(0.4*numTrajs) ;
	int numPts_perTraj = x_samples.cols();
	Eigen::Vector3d Qpt; 
	 
	double distance ;
	std::string path_to_weights = "/home/sudarshan/Planner_ws/src/rp-vio/rpvio_estimator/src/weight.csv";

	// std::cout << " Here " << std::endl;	

	MMDF.assign_weights(path_to_weights); 
	double mmd_cost;
	Eigen::MatrixXd TopIndices( num_top_samples , 1 );

	Eigen::MatrixXd MMDCostPerTraj( numTrajs , 1); 
	std::vector<double> MMD_cost;
	double DistMin; 
	int numDistanceSamples = 100; 

	Eigen::MatrixXd temp_x_samples;
	Eigen::MatrixXd temp_y_samples;
	Eigen::MatrixXd temp_z_samples; 

	Eigen::Vector3d PrevPt; 
	int numSamples_normal_uncertainity = 100 ; 	

	numDegreeSamples =  numSamples_normal_uncertainity ; 
	numDistanceSamples =  numSamples_normal_uncertainity ;

	for(int i =0 ; i< num_iterations ; i++ ){

		std::cout<<i<<std::endl;
		MMD_cost.clear();		 

		MMDCostPerTraj = Eigen::MatrixXd::Zero( numTrajs , 1);

		for( int j =0 ; j < numTrajs ; j++){
			double costPerTraj =0 ;
			double TrajLength =0 ; 

			for(int k =0; k<numPts_perTraj ; k +=5  ){

				Qpt << x_samples( j , k ) , y_samples( j , k ) , z_samples(j , k); 
				// distance  = MeasureSphericalDistance( Centers ,  Qpt); 

				distance = GetPlaneCollisionDistance( cuboids , Qpt );


				if(k ==0 ){
					PrevPt = Qpt; 
				}

				if(k > 0 ){
					TrajLength += ( Qpt - PrevPt ).norm(); 
					PrevPt = Qpt; 
				}


				if( distance < 2.0 ){
					mmd_cost =   MMDwithNormalUncertainity( cuboids , Qpt , numSamples_normal_uncertainity , false) ;  // MMDCost( distance , numDistanceSamples );
					
					costPerTraj += mmd_cost ;

				}
				else{
					mmd_cost = 0 ;
					costPerTraj += mmd_cost ;
				}

			}


			MMDCostPerTraj(j,0) = costPerTraj + 4*TrajLength ;
			MMD_cost.push_back(costPerTraj);

		}


		TopIndices = GetIndex( MMD_cost , num_top_samples );

		std::cout << " Got index " << std::endl;

		MeanTraj = UpdateMeanTraj( TopIndices , x_samples  ,  y_samples , z_samples , num_top_samples , numPts_perTraj );

		

		Eigen::Vector3d Q_mean; 
		double mean_mmd= 0 ; 
		double val =0;






		temp_x_samples = MeanTraj.at(0); 
		temp_y_samples = MeanTraj.at(1);
		temp_z_samples = MeanTraj.at(2);
		int numPts_mean_traj = temp_z_samples.cols() ;  

		std::vector<double> MMD_cost_vector ; 
		double mmd_point =0 ;

		for( int i =0 ; i<numPts_mean_traj ; i++ ){

			Q_mean  <<  temp_x_samples( 0 , i ) , temp_y_samples(0, i) ,temp_z_samples(0, i)  ;
			mmd_point = MMDwithNormalUncertainity( cuboids , Q_mean , numSamples_normal_uncertainity , true ) ; 
			MMD_cost_vector.push_back(mmd_point);
			mean_mmd += mmd_point;

			// std::cout<< val << std::endl ;

		}

		std::cout << mean_mmd <<  "  iter number " <<  i << std::endl ;

		std::sort(MMD_cost_vector.begin(), MMD_cost_vector.end());
		double top_cost =0 ; 

		for( int i=int(0.5*MMD_cost_vector.size() ) ; i < int(0.8*MMD_cost_vector.size() ) ; i++ ){
			top_cost += MMD_cost_vector.at(i);
		}
		std::cout << top_cost <<  "  top cost  iter number " <<  i << std::endl ;



				std::cout << " Got Mean " << std::endl;



		temp_x_samples(0 , 0 )=  x_samples(0,0) ; 
		temp_y_samples(0,0) =   y_samples(0,0) ; 
		temp_z_samples(0 ,0 ) =  z_samples( 0,0 ) ; 
/*
		temp_x_samples(0, numPts_perTraj -1 ) = x_samples( 0,numPts_perTraj-1 );
		temp_y_samples(0, numPts_perTraj -1 ) = y_samples(0 , numPts_perTraj -1);
		temp_z_samples(0, numPts_perTraj -1 ) = z_samples(0, numPts_perTraj -1) ;
*/

		VisOptimTraj( temp_x_samples ,  temp_y_samples  , temp_z_samples  ,  OptimTraj , CEM_MeanTrajectory);
				// std::cout << " Got vis " << std::endl;


		std::vector<Eigen::MatrixXd> Samples;

		Samples = PerturbTraj(  temp_x_samples , temp_y_samples , temp_z_samples ,  numPts_perTraj ,  SampleTraj , true  , numTrajs  , i  );

		x_samples = Samples.at(0);
		y_samples = Samples.at(1);
		z_samples = Samples.at(2); 


	}

	CEMOptimizedTraj.push_back( temp_x_samples ); 
	CEMOptimizedTraj.push_back( temp_y_samples );
	CEMOptimizedTraj.push_back(temp_z_samples ); 


	return CEMOptimizedTraj ;



}



/*

int Optimization::TrajectoryOptimization::CEM( Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , 
	Eigen::MatrixXd PlaneParams , ros::Publisher OptimTraj , ros::Publisher SampleTraj )
{

	std::vector<Eigen::MatrixXd> MeanTraj; 


	int numTrajs = x_samples.rows();
	int num_top_samples = int(0.1*numTrajs) ;
	int numPts_perTraj = x_samples.cols();
	Eigen::Vector3d Qpt; 
	int numNormals = PlaneParams.rows() ; 
	Eigen::MatrixXd Dist(numNormals, 1); 
	double distance ;
	std::string path_to_weights = "/home/sudarshan/PlanarPlanner_ws/src/CCO_VOXEL/include/CCO_VOXEL/weight.csv";

	MMDF.assign_weights(path_to_weights); 
	double mmd_cost;
	Eigen::MatrixXd TopIndices( num_top_samples , 1 );

	Eigen::MatrixXd MMDCostPerTraj( numTrajs , 1); 
	std::vector<double> MMD_cost;
	double DistMin; 
	int numDistanceSamples = 100; 


	Eigen::MatrixXd temp_x_samples;
	Eigen::MatrixXd temp_y_samples;
	Eigen::MatrixXd temp_z_samples; 

	Eigen::Vector3d PrevPt; 

	for(int i =0 ; i< num_iterations ; i++ ){

		std::cout<<i<<std::endl;
		MMD_cost.clear();		 

		MMDCostPerTraj = Eigen::MatrixXd::Zero( numTrajs , 1);

		for( int j =0 ; j < numTrajs ; j++){
			double costPerTraj =0 ;
			double TrajLength =0 ; 

			for(int k =0; k<numPts_perTraj ; k +=5  ){

				Qpt << x_samples( j , k ) , y_samples( j , k ) , z_samples(j , k); 
				Dist = MeasureDistance(PlaneParams ,  Qpt); 
				Eigen::MatrixXd::Index minRow, minCol;
				distance  = Dist.minCoeff(&minRow, &minCol);
				// distance = Dist.array().min(); 
				// std::cout<< distance << std::endl;

				// if( distance < 2.0 )
				// {
				// 	mmd_cost = MMDCost(distance , 100 );
				// 	costPerTraj += mmd_cost; 

				// 	std::cout <<costPerTraj << std::endl;

				// }

				if(k ==0 ){
					PrevPt = Qpt; 
				}

				if(k > 0 ){
					TrajLength += ( Qpt - PrevPt ).norm(); 
					PrevPt = Qpt; 
				}


				

				for( int n =0 ; n < numNormals ; n++){
					distance = Dist(n); 
					if( distance < 2.0   ){

						mmd_cost = MMDCost(distance , numDistanceSamples );
						costPerTraj += mmd_cost ; 
					}
				}
				

			}


			MMDCostPerTraj(j,0) = 2*costPerTraj + TrajLength ;
			MMD_cost.push_back(costPerTraj);

		}


		TopIndices = GetIndex( MMD_cost , num_top_samples );

		MeanTraj = UpdateMeanTraj( TopIndices , x_samples  ,  y_samples , z_samples , num_top_samples , numPts_perTraj );

		temp_x_samples = MeanTraj.at(0); 
		temp_y_samples = MeanTraj.at(1);
		temp_z_samples = MeanTraj.at(2); 


		temp_x_samples(0 , 0 )= 0 ; 
		temp_y_samples(0,0) = 0; 
		temp_z_samples(0 ,0 ) =0 ; 

		temp_x_samples(0, 74) = 5;
		temp_y_samples(0, 74) = 5;
		temp_z_samples(0, 74) = 5 ;

		VisOptimTraj( temp_x_samples ,  temp_y_samples  , temp_z_samples  ,  OptimTraj);

		std::vector<Eigen::MatrixXd> Samples;

		Samples = PerturbTraj(  temp_x_samples , temp_y_samples , temp_z_samples ,  numPts_perTraj ,  SampleTraj , true  , numTrajs  , i  );

		x_samples = Samples.at(0);
		y_samples = Samples.at(1);
		z_samples = Samples.at(2); 


	}





}

*/