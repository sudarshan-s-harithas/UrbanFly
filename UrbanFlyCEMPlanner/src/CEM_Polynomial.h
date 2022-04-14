
#include"utils.h"
#include"bernstein.h"
// #include"bsplineNonUnif.h"
// #include"Map.h"
#include<random>
#include<algorithm>
#include"visulization.h"
#include <fstream>
#include <experimental/filesystem>
#include"MMD.h"

MMDFunctions::MMD_variants MMDF2;
Visualizer::Trajectory_visualizer traj_vis;


namespace PolynomialFormulation{

	class CEMPolynomialFormulation{

	public:

		std::vector<Eigen::MatrixXd> CrossEntropyOptimize_Polynomial( Bernstein::BernsteinPath bTraj,   Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , std::vector<Eigen::Vector3d>  Centers,  
		ros::Publisher OptimTraj , ros::Publisher SampleTraj , std::vector<CuboidObject> cuboids , ros::Publisher  CEM_MeanTrajectory  );
		double MMDwithNormalUncertainity( std::vector<CuboidObject> cuboids , Eigen::Vector3d Qpt , int numSamples_normal_uncertainity , bool Ismean );

		void SetUncertainityMeasurments();
		double costPerTrajectory(  std::vector<Eigen::Vector3d> traj, std::vector<Eigen::Vector3d> trajAcc, std::vector<Eigen::Vector3d> initBernsteinTraj,  bool is_mean , std::vector<CuboidObject> cuboids);
		double get_elastic_band_cost( std::vector<Eigen::Vector3d> traj_in ) ;
		double get_variance( Eigen::MatrixXd one_dimension_trajectory , int iter ,  Eigen::MatrixXd MeanCoeff) ;
		double get_acc_cost( Eigen::Vector3d acc_in );
	private:

		int numDegreeSamples  = 100 ; 
		int  numDistanceSamples  = 100;
		Eigen::MatrixXd NormalDistribution ; 
		Eigen::MatrixXd DistanceDistribution ;
		int iter_cnt = 0 ; 

		float execTime = 2.0 ; 

		int numSampleTrajs = 100 ; 

		int ptsPerTraj = 50 ; 
		int numSamples_normal_uncertainity = 100 ;

		inline Eigen::MatrixXd convertVecTrajToMatTraj(std::vector<Eigen::Vector3d> arr);
		inline std::vector<Eigen::Vector3d> convertMatTrajToVecTraj(Eigen::MatrixXd mat);

		Eigen::MatrixXd optimTrajCoeffs = Eigen::MatrixXd::Zero(11,3);
		Eigen::Vector3d var_vector;
		int topSamples     = 25;
		bool PubTraj = true ;

	};


}



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


void PolynomialFormulation::CEMPolynomialFormulation::SetUncertainityMeasurments()
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

double PolynomialFormulation::CEMPolynomialFormulation::MMDwithNormalUncertainity( std::vector<CuboidObject> cuboids , Eigen::Vector3d Qpt , int numSamples_normal_uncertainity  ,
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

	radius = Eigen::MatrixXf::Constant( 1, numSamples_normal_uncertainity,  1.5); 

    Eigen:Vector3d Qpt_transformed ;
    float distance_ ;



    Eigen::MatrixXf Distance_Measurments( 1 , numSamples_normal_uncertainity )  ;

    std::vector<Eigen::Vector3d> vertices_cuboid ;
    double val ; 

	for( int i=0 ; i < cuboids.size() ; i++){

		vertices_cuboid = cuboids[i].vertices_ ; 
		Center_per_cuboid = DetermineCenter( vertices_cuboid ) ;
		Q_Center_distance =  ( Center_per_cuboid - Qpt ).norm() ;

		if( Q_Center_distance > 2){
			continue ;
		}

		for( int i =0 ; i < numSamples_normal_uncertainity ; i++ ){

			Qpt_transformed = TransformQpt(  Qpt , NormalDistribution( 0, i) , theta , Center_per_cuboid );
			distance_ =  float ( MeasureSDFCollision( Qpt_transformed , DistanceDistribution.row(i)  ) );
			Distance_Measurments( 0 , i ) = radius( 0, i) - distance_ ;
		}

		actual_distribution = zero_matrix.cwiseMax( Distance_Measurments ) ;
		val = MMDF2.MMD_transformed_features_RBF(actual_distribution) ;
		MMD_cost += val ; 

	}



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




std::vector<Eigen::MatrixXd> PolynomialFormulation::CEMPolynomialFormulation::CrossEntropyOptimize_Polynomial( Bernstein::BernsteinPath bTraj,   Eigen::MatrixXd x_samples ,  Eigen::MatrixXd y_samples  , Eigen::MatrixXd z_samples , int num_iterations , std::vector<Eigen::Vector3d>  Centers,  
		ros::Publisher OptimTraj , ros::Publisher SampleTraj , std::vector<CuboidObject> cuboids , ros::Publisher  CEM_MeanTrajectory  )
{

	SetUncertainityMeasurments();
	std::string path_to_weights = "/home/sudarshan/Planner_ws/src/rp-vio/rpvio_estimator/src/weight.csv";


	MMDF2.assign_weights(path_to_weights); 

	std::vector<Eigen::Vector3d> wayPts ; 

	Eigen::Vector3d Qpt1 ; 

	for( int i= 0 ; i <x_samples.cols() ; i++  ){

		Qpt1 << x_samples( 0 , i ) , y_samples( 0, i ) , z_samples( 0 , i ); 
		wayPts.push_back( Qpt1 );
	}

	std::cout<<"----Generating bernstein trajectory for "<<wayPts.size()<<" points"<<std::endl;
	std::vector<Eigen::Vector3d> prev_mean_bernstein_trajectory; 
	bTraj.generateCoeffMatrices(wayPts.size(), execTime);
	bTraj.generateTrajCoeffs(wayPts);                                            // this would initialize the trajectory coefficients with those of the fast planner
	bTraj.generateCoeffMatrices(ptsPerTraj, execTime);



    Eigen::MatrixXd initCoeff = convertVecTrajToMatTraj(bTraj.coeffs);           // this takes in a vector of 11 indices and returns a matrix of size ptsPerTrajx3

    Eigen::MatrixXd initWayPts = (bTraj.P)*initCoeff;                            // this returns a 50x3 matrix

    std::vector<Eigen::Vector3d> initBernsteinTraj = convertMatTrajToVecTraj(initWayPts); // returns initial trajectory 

    std::vector<Eigen::Vector3d> coeffs_ = bTraj.coeffs;                         // initial coefficients

    var_vector = Eigen::Vector3d::Zero() ;

    std::vector<Eigen::Vector3d> temp_mean_bernstein_trajectory ; 





    double var  = 5;
    var_vector.x() = 0.1;
    var_vector.y() = 0.1;
    var_vector.z() = 0.1;

    int num_prev_top_traj = topSamples; 

    Eigen::MatrixXd prev_TopX( num_prev_top_traj, ptsPerTraj  );
    Eigen::MatrixXd prev_TopY( num_prev_top_traj , ptsPerTraj  );
    Eigen::MatrixXd prev_TopZ( num_prev_top_traj , ptsPerTraj  );

    std::vector<Eigen::Vector3d> mean_bernstein_trajectory ;
    mean_bernstein_trajectory = initBernsteinTraj;





    for(int iter = 0; iter<num_iterations; iter++)
    {

    	std::vector<std::vector<Eigen::Vector3d> > trajs;
    	std::vector<std::vector<Eigen::Vector3d> > trajsAcc;
        std::cout<<"Cross entropy Iteration "<<iter<<std::endl;
        std::vector<Eigen::MatrixXd> perturbedCoeffs = bTraj.generatePerturbedCoeffs(numSampleTrajs, coeffs_, var_vector);

        Eigen::MatrixXd xPts = ((bTraj.P)*(perturbedCoeffs.at(0).transpose())).transpose();
        Eigen::MatrixXd yPts = ((bTraj.P)*(perturbedCoeffs.at(1).transpose())).transpose();
        Eigen::MatrixXd zPts = ((bTraj.P)*(perturbedCoeffs.at(2).transpose())).transpose();


        Eigen::MatrixXd xAccPts = ((bTraj.Pddot)*(perturbedCoeffs.at(0).transpose())).transpose();
        Eigen::MatrixXd yAccPts = ((bTraj.Pddot)*(perturbedCoeffs.at(1).transpose())).transpose();
        Eigen::MatrixXd zAccPts = ((bTraj.Pddot)*(perturbedCoeffs.at(2).transpose())).transpose();


        std::vector<double> costTrajs(numSampleTrajs);

        for(int i = 0; i<numSampleTrajs; i++)
        {

            std::vector<Eigen::Vector3d> traj;
            std::vector<Eigen::Vector3d> trajAcc;
            bool getCost = true;
            for(int j = 0; j<ptsPerTraj; j++)
            {

            Eigen::Vector3d pt(xPts(i,j), yPts(i,j), zPts(i,j));
            Eigen::Vector3d ptAcc(xAccPts(i,j), yAccPts(i,j), zAccPts(i,j));

            traj.push_back(pt);
            trajAcc.push_back(ptAcc);          
            }

            // trajs.push_back(traj);
            // trajsAcc.push_back(trajAcc);

            bool is_mean = false;

            double cost = costPerTrajectory(traj, trajAcc, initBernsteinTraj,  is_mean , cuboids);
            costTrajs.at(i) = cost;
            traj.clear();
            trajAcc.clear() ;

           
        }


    std::vector<double> costTrajsorted = costTrajs;
    std::sort(costTrajsorted.begin(), costTrajsorted.end());

    double valOptim = costTrajsorted.at(0);
    auto itrOptim   = std::find(costTrajs.begin(), costTrajs.end(), valOptim);
    int indexOptim  = itrOptim - costTrajs.begin();


    for(int k = 0; k<11; k++)
    {
        optimTrajCoeffs(k,0) = perturbedCoeffs.at(0)(indexOptim,k);//coeffs_.at(k)(0);//perturbedCoeffs.at(0)(indexOptim,k);// // // 
        optimTrajCoeffs(k,1) = perturbedCoeffs.at(1)(indexOptim,k);//coeffs_.at(k)(1);//perturbedCoeffs.at(1)(indexOptim,k);// // //;
        optimTrajCoeffs(k,2) = perturbedCoeffs.at(2)(indexOptim,k);//coeffs_.at(k)(2);//perturbedCoeffs.at(2)(indexOptim,k);// ///
    }

    std::vector<Eigen::Vector3d> newCoeffs(11);

    std::vector<int> topIndexes;

/*
    if( iter > 0)
    {

    	for(int index_top = 0; index_top < num_prev_top_traj ; index_top++ ){

    		xPts.row(index_top)  = prev_TopX.row(index_top);
    		yPts.row(index_top)  = prev_TopY.row(index_top);
    		zPts.row(index_top)  = prev_TopZ.row(index_top);

    	}
    }
*/
    topIndexes.clear();
    for(int i = 0; i<topSamples; i++)
    {
        auto it = costTrajsorted.begin();
        double costVal = *it;
        auto itr = std::find(costTrajs.begin(), costTrajs.end(), costVal);
        int index = itr - costTrajs.begin();
        topIndexes.push_back(index);

        costTrajsorted.erase(costTrajsorted.begin());
    }


    Eigen::MatrixXd sum_top_X = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd sum_top_Y = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd sum_top_Z = Eigen::MatrixXd::Zero(1,11);

    Eigen::MatrixXd diffTopCoeffsX = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd diffTopCoeffsY = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd diffTopCoeffsZ = Eigen::MatrixXd::Zero(1,11);


    Eigen::MatrixXd xCoeff = perturbedCoeffs.at(0);
    Eigen::MatrixXd yCoeff = perturbedCoeffs.at(1);
    Eigen::MatrixXd zCoeff = perturbedCoeffs.at(2);



    Eigen::MatrixXd TopX( topSamples , ptsPerTraj  );
    Eigen::MatrixXd TopY( topSamples , ptsPerTraj  );
    Eigen::MatrixXd TopZ( topSamples , ptsPerTraj  );

    Eigen::MatrixXd TopX_mean( 1 , ptsPerTraj  );
    Eigen::MatrixXd TopY_mean( 1 , ptsPerTraj  );
    Eigen::MatrixXd TopZ_mean( 1 , ptsPerTraj  );

    TopX_mean = Eigen::MatrixXd::Zero( 1 , ptsPerTraj) ; 
    TopY_mean = Eigen::MatrixXd::Zero(1, ptsPerTraj ) ;
    TopZ_mean = Eigen::MatrixXd::Zero(1 ,ptsPerTraj) ; 


    int index = 0;

    for( int p =0 ; p < topSamples ; p ++)
    {
    TopX.row(p)= xPts.row( topIndexes.at(p));  // xPts dimension numsamples x pointspertraj 
    TopY.row(p)= yPts.row( topIndexes.at(p));
    TopZ.row(p)= zPts.row( topIndexes.at(p));

/*    if( p < num_prev_top_traj){

    	prev_TopX.row(p) = TopX.row(p);
    	prev_TopY.row(p) = TopY.row(p);
    	prev_TopZ.row(p) = TopZ.row(p);
    }*/
	} 


	if( PubTraj == true)
	{

    traj_vis.visulize_sampled_trajectories( TopX , TopY , TopZ , topSamples , ptsPerTraj,  SampleTraj);
    }   

    // std::vector<Eigen::Vector3d> temp_mean_bernstein_trajectory ; 

    for( int col =0 ; col < ptsPerTraj ; col++)
    {
        Eigen::Vector3d waypoint_mean; 
    for( int row= 0 ; row < topSamples ; row++){

            TopX_mean(0,col) += TopX( row, col); 
            TopY_mean(0,col) += TopY( row, col); 
            TopZ_mean(0,col) += TopZ( row, col); 
        }

        TopX_mean /= topSamples;
        TopY_mean /= topSamples;
        TopZ_mean /= topSamples;


        waypoint_mean(0) =    TopX_mean(0,col) ;
        waypoint_mean(1) =  TopY_mean(0,col)  ;
        waypoint_mean(2) =   TopZ_mean(0,col);



    
        temp_mean_bernstein_trajectory.push_back(waypoint_mean);

    }

    Eigen::MatrixXd sumTopCoeffsX = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd sumTopCoeffsY = Eigen::MatrixXd::Zero(1,11);
    Eigen::MatrixXd sumTopCoeffsZ = Eigen::MatrixXd::Zero(1,11);

    Eigen::MatrixXd Coeffx_top_Set =  Eigen::MatrixXd::Zero( topSamples , 11  );
    Eigen::MatrixXd Coeffy_top_Set = Eigen::MatrixXd::Zero( topSamples , 11  );
    Eigen::MatrixXd Coeffz_top_Set = Eigen::MatrixXd::Zero( topSamples , 11  );


    for(int i = 0; i<topSamples; i++)
    {   
        // std::cout<<xCoeff.row(0)<<std::endl;
        
        int index_ = topIndexes.at(i);

        sumTopCoeffsX += xCoeff.row(index_); 
        sumTopCoeffsY += yCoeff.row(index_);
        sumTopCoeffsZ += zCoeff.row(index_);

        Coeffx_top_Set.row(i) =  xCoeff.row(index_)/xCoeff.row(index_).norm() ; 
        Coeffy_top_Set.row(i) = yCoeff.row(index_) / yCoeff.row(index_).norm(); 
        Coeffz_top_Set.row(i) = zCoeff.row(index_)/ zCoeff.row(index_).norm() ;

    }     

    Eigen::MatrixXd NewUpdateCoeffs( 3, 11 ); 

    Eigen::Vector3d Coeff_updated;    

    for(int j = 0; j<11; j++)
    {
    	newCoeffs.at(j)(0) = double(sumTopCoeffsX(0,j) /  sumTopCoeffsX.row(0).norm() ) ; 
    	newCoeffs.at(j)(1) = double(sumTopCoeffsY(0,j) / sumTopCoeffsY.row(0).norm() ); 
    	newCoeffs.at(j)(2) = double(sumTopCoeffsZ(0,j) /sumTopCoeffsZ.row(0).norm() );

    	NewUpdateCoeffs( 0, j)  =newCoeffs.at(j)(0) ;  
    	NewUpdateCoeffs( 1 , j)  = newCoeffs.at(j)(1) ;  
    	NewUpdateCoeffs( 2, j)  = newCoeffs.at(j)(2)  ;

    }

    NewUpdateCoeffs.row(0) /= NewUpdateCoeffs.row(0).norm(); 
    NewUpdateCoeffs.row(1) /= NewUpdateCoeffs.row(1).norm(); 
    NewUpdateCoeffs.row(2) /= NewUpdateCoeffs.row(2).norm(); 


    std::vector<Eigen::Vector3d> Final;
    for(int i =0 ;i < 11 ; i++){
    	Coeff_updated << NewUpdateCoeffs(0 , i) , NewUpdateCoeffs(1 , i) , NewUpdateCoeffs(2 , i) ;  
    	Final.push_back(Coeff_updated); 
    }

    var_vector.x() = get_variance(Coeffx_top_Set , iter  , NewUpdateCoeffs.row(0) ); 
    var_vector.y() = get_variance(Coeffy_top_Set, iter , NewUpdateCoeffs.row(1) );
    var_vector.z() = get_variance(Coeffz_top_Set , iter , NewUpdateCoeffs.row(2) );

    std::cout << var_vector.x()  <<   " " << var_vector.y()  << " +++ " << var_vector.z()   <<  "  " << "updated variance" << std::endl;
    


    bTraj.generateTrajCoeffs(mean_bernstein_trajectory);

    coeffs_.clear(); 
    coeffs_ = bTraj.coeffs;

    // bTraj.coeffs = Final ;
    // coeffs_.clear(); 
    // coeffs_ = Final ;


    std::vector<Eigen::Vector3d> mean_trajAcc;

    Eigen::Vector3d Pt_Acc; 
    for( int i =0 ; i < ptsPerTraj ; i++){

    	Eigen::Vector3d  Pt_Acc( 0 , 0 , 0 );
    	mean_trajAcc.push_back(Pt_Acc);
    }

    bool is_mean = true ;

    }


    Eigen::MatrixXd bestTraj = ((bTraj.P) * optimTrajCoeffs);

    std::vector<Eigen::MatrixXd> CEMOptimizedTraj; 

    std::cout << bestTraj.rows() << " ******* " << bestTraj.cols() <<  std::endl  ;

    Eigen::MatrixXd X_Optimized( 1,  bestTraj.rows() );
    Eigen::MatrixXd Y_Optimized( 1, bestTraj.rows());
    Eigen::MatrixXd Z_Optimized( 1, bestTraj.rows() );

    for(int ind = 0; ind < bestTraj.rows(); ind++)
    {
        Eigen::Vector3d pt;

        pt(0) = bestTraj(ind, 0);
        pt(1) = bestTraj(ind, 1);
        pt(2) = bestTraj(ind, 2);

        X_Optimized( 0 , ind ) = bestTraj(ind, 0);
        Y_Optimized(0, ind ) = bestTraj(ind, 1);
        Z_Optimized(0, ind) = bestTraj(ind , 2) ;

        // CEMOptimizedTraj.push_back(pt);

        // std::cout<<pt(0)<<"\t"<<pt(1)<<"\t"<<pt(2)<<std::endl;
    }

    CEMOptimizedTraj.push_back(X_Optimized);
    CEMOptimizedTraj.push_back(Y_Optimized);
    CEMOptimizedTraj.push_back(Z_Optimized);

    // std::cout << smoothness_cost << "smoothness_cost" <<std::endl;

    return CEMOptimizedTraj ;
}



double PolynomialFormulation::CEMPolynomialFormulation::costPerTrajectory(  std::vector<Eigen::Vector3d> traj, std::vector<Eigen::Vector3d> trajAcc, std::vector<Eigen::Vector3d> initBernsteinTraj, bool is_mean , std::vector<CuboidObject> cuboids)
{

    double cost = 0.0;
    double collisionCost  = 0.0;
    double stabilityCost  = 0.0;
    double smoothnessCost = 0.0;
    double elastic_band_cost= 0; 
    int number_of_points_in_distribution =100;

    elastic_band_cost = get_elastic_band_cost(traj);

    Eigen::Vector3d WayPoint ; 
    bool Ismean = false ;


    for(int i = 0; i<traj.size(); i++)
    {

    	Eigen::Vector3d pt = traj.at(i);
    	Eigen::Vector3d ptAcc = trajAcc.at(i);
    	Eigen::Vector3d ptInit = initBernsteinTraj.at(i);

    	stabilityCost  +=   get_acc_cost(ptAcc);  // ptAcc.norm();
    	smoothnessCost += (pt - ptInit).norm();

    	collisionCost += MMDwithNormalUncertainity( cuboids ,pt , numSamples_normal_uncertainity ,  Ismean  ) ; 
    }  

    cost = collisionCost  + 0.50*stabilityCost  + 0.001*elastic_band_cost;

    return cost ;


}


double PolynomialFormulation::CEMPolynomialFormulation::get_elastic_band_cost( std::vector<Eigen::Vector3d> traj_in )
{
 double elastic_cost=0 ; 
for(int i = 1; i<traj_in.size()-1; i++)
    {
        Eigen::Vector3d pt_before = traj_in.at(i-1);
        Eigen::Vector3d pt_current = traj_in.at(i);
        Eigen::Vector3d pt_next = traj_in.at(i+1);

        Eigen::Vector3d diff; 

        diff = pt_before - 2*pt_current + pt_next ; 
        elastic_cost += diff.norm();

}

return elastic_cost;

}



inline Eigen::MatrixXd PolynomialFormulation::CEMPolynomialFormulation::convertVecTrajToMatTraj(std::vector<Eigen::Vector3d> arr)
{
    Eigen::MatrixXd trajPoints =  Eigen::MatrixXd::Zero(arr.size(), 3);
    
    for(int i = 0; i<arr.size(); i++)
    {
        trajPoints(i,0) = arr.at(i)(0);
        trajPoints(i,1) = arr.at(i)(1);
        trajPoints(i,2) = arr.at(i)(2);
    }

    return trajPoints;
}

inline std::vector<Eigen::Vector3d> PolynomialFormulation::CEMPolynomialFormulation::convertMatTrajToVecTraj(Eigen::MatrixXd mat)
{
    std::vector<Eigen::Vector3d> trajPoints;

    for(int i = 0; i<mat.rows(); i++)
    {
        Eigen::Vector3d pt;
        pt(0) = mat(i,0);
        pt(1) = mat(i,1);
        pt(2) = mat(i,2);

        trajPoints.push_back(pt);
    }

    return trajPoints;
}

/*
double PolynomialFormulation::CEMPolynomialFormulation::get_variance( Eigen::MatrixXd one_dimension_trajectory , int iter ,  Eigen::MatrixXd MeanCoeff  )
{

	int coeff_size = 11;  
	double variance_value=0;
	Eigen::MatrixXd mean(1,11);
	Eigen::MatrixXd mean_sum(1,1);
	mean_sum.setZero();


	int total_size = one_dimension_trajectory.rows()*one_dimension_trajectory.cols();

	for(int k=0 ; k< one_dimension_trajectory.cols() ; k++ ){
		for(int l=0 ; l< one_dimension_trajectory.rows() ; l++){
			mean_sum(0,0) += one_dimension_trajectory( l, k )/(total_size );
		}
	}

	for( int i=0 ;i < one_dimension_trajectory.rows() ; i++){
		for( int j =0; j <11 ; j++){
			variance_value += pow( ( one_dimension_trajectory( i,j) - mean_sum(0,0) ), 2);

		}
	}

	variance_value = variance_value/(total_size); 

	return variance_value ;

}
*/

double DetermineVar( Eigen::MatrixXd values )
{

	double variance_value = 0; 

	double mean =0 ;

	// std::cout << values.cols() << std::endl ;
	// std::cout << "-----------------" << std::endl ;

	for( int i =0 ;i <values.cols() ; i++ )
	{

		mean += values(0, i)/values.cols()  ;
	}


	for( int i =0 ;i <values.cols() ; i++ )
	{
		variance_value += pow( ( values( 0,i) - mean ), 2);
	}

	return variance_value  ; 
}

double PolynomialFormulation::CEMPolynomialFormulation::get_variance( Eigen::MatrixXd Trajectories , int iter ,  Eigen::MatrixXd MeanCoeff  )
{


	int coeff_size = 11 ; 
	double variance_value =0 ; 
	double temp_var = 0;


	int numTrajs = Trajectories.rows();

	Eigen::MatrixXd CoeffVals( 1, numTrajs );

	for( int j =0 ; j < Trajectories.cols() ; j ++ ){

		for( int i=0 ; i < numTrajs ; i++ ){
			CoeffVals( 0, i)=  Trajectories( i, j );

		}
		temp_var += DetermineVar(CoeffVals  );

	}

	return temp_var   ;

}


double PolynomialFormulation::CEMPolynomialFormulation::get_acc_cost( Eigen::Vector3d acc_in ){

double acc_val = acc_in.norm();
double amin = -1.0 ; 
double amax = 1.5 ;

double acc_cost =0 ;

if( acc_val > amax || acc_val < amin){

acc_cost = ( acc_val - amin)*(acc_val - amax);
}
else{
    acc_cost =0 ;
}

return acc_cost;
}