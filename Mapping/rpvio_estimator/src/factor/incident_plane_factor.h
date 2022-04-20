#pragma once
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

struct IncidentPlaneFactor
{
    IncidentPlaneFactor(const Eigen::Vector3d &pt_i) : _pt_i(pt_i) {}

    template <typename T>
        bool operator()(
            const T* const para_n,
            const T* const para_depth,
            const T* const inv_depth_i,
            const T* const para_Pose_i,
            const T* const para_ExPose,
            T* residual
        ) const {
            Eigen::Matrix<T, 3, 1> Pi(para_Pose_i[0], para_Pose_i[1], para_Pose_i[2]);
            Eigen::Quaternion<T> Qi(para_Pose_i[6], para_Pose_i[3], para_Pose_i[4], para_Pose_i[5]);

            Eigen::Matrix<T, 3, 1> tic(para_ExPose[0], para_ExPose[1], para_ExPose[2]);
            Eigen::Quaternion<T> qic(para_ExPose[6], para_ExPose[3], para_ExPose[4], para_ExPose[5]);

            Eigen::Matrix<T, 3, 1> pts_camera_i = _pt_i.cast<T>() / inv_depth_i[0];
            Eigen::Matrix<T, 3, 1> pts_imu_i = qic * pts_camera_i + tic;
            Eigen::Matrix<T, 3, 1> pts_w = Qi * pts_imu_i + Pi;
            Eigen::Matrix<T, 3, 1> pts_c = qic.inverse() * (pts_w - tic);

            Eigen::Matrix<T, 4, 1> plane;
            plane << para_n[0], para_n[1], para_n[2], -para_depth[0];

            residual[0] = plane.dot(pts_c.homogeneous());
            return true;
        }
    
    static ceres::CostFunction* Create(const Eigen::Vector3d &pt_i)
    {
        return (new ceres::AutoDiffCostFunction<IncidentPlaneFactor, 1, 3, 1, 1, 7, 7>
                (new IncidentPlaneFactor(pt_i)));
    }

    Eigen::Vector3d _pt_i;
};