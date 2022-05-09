#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "eigenmvn.h"
#include <string.h>

using Eigen::MatrixXd;
using namespace Eigen;
using namespace std;

MatrixXd diff(MatrixXd input)
{
  MatrixXd output;
  output = MatrixXd::Zero(input.rows()-1, input.cols());

  output = input.block(1, 0, input.rows()-1, input.cols()) - input.block(0, 0, input.rows()-1, input.cols());

  return output;
}

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;

  MatrixXd I = MatrixXd::Identity(4, 4);
  std::cout << I << std::endl;
  std::cout << diff(I) << std::endl;
  std::cout << diff(diff(I)) << std::endl;
  std::cout << diff(diff(diff(I))) << std::endl;

  /**
  * Write 100 trajectories to file !
  */

  int num_goal = 1000;
  int num = 70;

  double x_init =  30.0;
  double y_init =  -5.0;
  double z_init =  0.0;

  double x_des_traj_init = x_init;
  double y_des_traj_init = y_init;
  double z_des_traj_init = z_init;

  double vx_des = 1.0;
  double vy_des = -0.70;
  double vz_des = -0.2;

  // ################################# Hyperparameters
  double t_fin = 75;

  // ################################# noise sampling

  // ########### Random samples for batch initialization of heading angles
  MatrixXd identity = MatrixXd::Identity(num, num);
  MatrixXd A = diff(diff(identity));

  std::cout << "A matrix size is " << to_string(A.rows()) << ", " << to_string(A.cols()) << std::endl;
  
  MatrixXd temp_1 = MatrixXd::Zero(1, num);
  MatrixXd temp_2 = MatrixXd::Zero(1, num);
  MatrixXd temp_3 = MatrixXd::Zero(1, num);
  MatrixXd temp_4 = MatrixXd::Zero(1, num);

  temp_1(0, 0) = 1.0;
  temp_2(0, 0) = -2;
  temp_2(0, 1) = 1;
  temp_3(0, num-1) = -2;
  temp_3(0, num-2) = 1;
  temp_4(0, num-1) = 1.0;

  MatrixXd A_mat = MatrixXd::Zero(num+2, num);
  A_mat.row(0) = temp_1;
  A_mat.row(1) = temp_2;
  A_mat.block(2, 0, A.rows(), A.cols()) = A;
  A_mat.row(A.rows()+2) = temp_3;
  A_mat.row(A.rows()+3) = temp_4;

  A_mat = -A_mat;
  std::cout << "A_mat matrix size is " << to_string(A_mat.rows()) << ", " << to_string(A_mat.cols()) << std::endl;

  MatrixXd R = A_mat.transpose() * A_mat;
  std::cout << "R matrix size is " << to_string(R.rows()) << ", " << to_string(R.cols()) << std::endl;

  MatrixXd mu = MatrixXd::Zero(num, 1);
  MatrixXd cov = 0.03 * R.inverse();

  Eigen::EigenMultivariateNormal<double> normX_solver(mu, cov);
  Eigen::EigenMultivariateNormal<double> normY_solver(mu, cov);
  Eigen::EigenMultivariateNormal<double> normZ_solver(mu, cov);

  // ################# Gaussian Trajectory Sampling
  MatrixXd eps_kx = normX_solver.samples(num_goal).transpose();
  MatrixXd eps_ky = normY_solver.samples(num_goal).transpose();
  MatrixXd eps_kz = normZ_solver.samples(num_goal).transpose();

  double x_fin = x_des_traj_init+vx_des*t_fin;
  double y_fin = y_des_traj_init+vy_des*t_fin;
  double z_fin = z_des_traj_init+vz_des*t_fin;

  VectorXd t_interp = VectorXd::LinSpaced(num, 0, t_fin);
  VectorXd x_interp = (x_des_traj_init + ((x_fin-x_des_traj_init)/t_fin) * t_interp.array()).matrix();
  VectorXd y_interp = (y_des_traj_init + ((y_fin-y_des_traj_init)/t_fin) * t_interp.array()).matrix();
  VectorXd z_interp = (z_des_traj_init + ((z_fin-z_des_traj_init)/t_fin) * t_interp.array()).matrix();

  MatrixXd x_samples(eps_kx.rows(), eps_kx.cols());
  MatrixXd y_samples(eps_kx.rows(), eps_kx.cols());
  MatrixXd z_samples(eps_kx.rows(), eps_kx.cols());
  
  x_samples = eps_kx;
  x_samples.rowwise() += x_interp.transpose();
  y_samples = eps_ky;
  y_samples.rowwise() += y_interp.transpose();
  z_samples = eps_kz;
  z_samples.rowwise() += z_interp.transpose();

  // z_samples.rowwise() = y_interp.transpose() + eps_kz.rowwise();
  std::cout << "samples matrix size is " << to_string(x_samples.rows()) << ", " << to_string(x_samples.cols()) << std::endl;

  std::ofstream file_samples("path_samples.txt");

  for (int i = 0; i < x_samples.rows(); i++)
  {
    for (int j = 0; j < x_samples.cols(); j++)
    {
      file_samples << x_samples(i, j) << " " << y_samples(i, j) << " " << z_samples(i, j) << std::endl;
    }
  }
  file_samples.close();

  return 0;
}
