#pragma once
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>

struct OrthogonalPlaneFactor
{
    template <typename T>
        bool operator()(
            const T* const normali_array,
            const T* const normalj_array,
            T* residual
        ) const {
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> normal_i(normali_array);
            Eigen::Map<const Eigen::Matrix<T, 3, 1>> normal_j(normalj_array);

            residual[0] = T(100) * normal_i.dot(normal_j);
            return true;
        }
    
    static ceres::CostFunction* Create()
    {
        return (new ceres::AutoDiffCostFunction<OrthogonalPlaneFactor, 1, 3, 3>
                (new OrthogonalPlaneFactor()));
    }
};