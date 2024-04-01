#ifndef _UKF_TYPES_H_
#define _UKF_TYPES_H_
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
using namespace Eigen;
namespace UKF {

template <typename scalar>
struct ImuData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double stamp;
    Eigen::Matrix<scalar, 3, 1> acc;
    Eigen::Matrix<scalar, 3, 1> gyr;
};

template <typename scalar>
struct UwbData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<scalar, 3, 1> data;
    Eigen::Matrix<scalar, 3, 3> cov;
};

struct STATE {
    Eigen::Matrix3d Rot;
    Eigen::Vector3d v, p, b_gyro, b_acc;
};
struct WEIGHTS {


    // template <typename T>
    struct W {
        double sqrt_d_lambda;
        double wj;
        double wm;
        double w0;

        W(double l, double alpha) {
            double m = (std::pow(alpha, 2) - 1) * l;
            sqrt_d_lambda = std::sqrt(l + m);
            wj = 1 / (2 * (l + m));
            wm = m / (l + m);
            w0 = m / (l + m) + 3 - std::pow(alpha, 2);
        }
    };
    
    W red_d;
    W q;
    W up_d;
    W aug_d;
    W aug_q;

    WEIGHTS(double red_d_val, double q_val, double up_d_val, double aug_d_val, double aug_q_val, double* alpha) :
    red_d(red_d_val, alpha[0]),
    q(q_val, alpha[1]),
    up_d(up_d_val, alpha[2]),
    aug_d(aug_d_val, alpha[3]),
    aug_q(aug_q_val, alpha[4]) {}
};
// template <typename T>
// struct W {
//     T sqrt_d_lambda;
//     T wj;
//     T wm;
//     T w0;

//     W(T l, T alpha) {
//         T m = (std::pow(alpha, 2) - 1) * l;
//         sqrt_d_lambda = std::sqrt(l + m);
//         wj = 1 / (2 * (l + m));
//         wm = m / (l + m);
//         w0 = m / (l + m) + 3 - std::pow(alpha, 2);
//     }
// };

// struct WEIGHTS {
//     W<double> red_d;
//     W<double> q;
//     W<double> up_d;
//     W<double> aug_d;
//     W<double> aug_q;

//     WEIGHTS(double red_d_val, double q_val, double up_d_val, double aug_d_val, double aug_q_val, double* alpha) :
//         red_d(red_d_val, alpha[0]),
//         q(q_val, alpha[1]),
//         up_d(up_d_val, alpha[2]),
//         aug_d(aug_d_val, alpha[3]),
//         aug_q(aug_q_val, alpha[4])
//     {}
// };

} // namespace UKF
#endif
