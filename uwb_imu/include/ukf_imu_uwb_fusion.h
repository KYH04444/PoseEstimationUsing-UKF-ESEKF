#ifndef _UKF_IMU_UWB_FUSION_H_
#define _UKF_IMU_UWB_FUSION_H_

#include <iostream>
#include <vector>
#include "ukf_types.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
using namespace std;
namespace UKF {

class ImuUwbFusionUkf {
public:
    ImuUwbFusionUkf();
    void imuInit(const vector<ImuData<double>> &imu_datas);
    void H_num(const Eigen::MatrixXd &uwb_data, const Eigen::VectorXi &idxs, const Eigen::MatrixXd &R);
    void update(const Eigen::MatrixXd &uwb_data, const Eigen::MatrixXd &R);
    void F_num(const vector<ImuData<double>> &imu_data, double dt);
    void G_num(const vector<ImuData<double>> &imu_data, double dt);
    void state_update();
    void cfgImuVar(double sigma_an, double sigma_wn, double sigma_aw, double sigma_ww);
    Eigen::Matrix3d _Exp(const Eigen::Vector3d &gyroDt);
    Eigen::Matrix3d _Log(const Eigen::Matrix3d &Rot);
    Eigen::Vector3d vee(const Eigen::Matrix3d &Phi);
    Eigen::Matrix3d wedge(const Eigen::Vector3d &v);
    STATE red_phi(const STATE &state, const Eigen::VectorXd &xi);
    STATE getState();
    STATE up_phi(const STATE &state, const Eigen::VectorXd &xi);
    Eigen::VectorXd phi_inv(const STATE &state, const STATE &hat_state);
    Eigen::Matrix3d rotationMatrixFromVectors(Eigen::Vector3d v0, Eigen::Vector3d v1);
    Eigen::Vector3d h(const STATE &State);

private:
    Eigen::Matrix3d Rot;
    Eigen::Vector3d v, p, b_gyro, b_acc;
    Eigen::Matrix<double, 12, 12> Q;
    Eigen::Vector5d alpha;
    Eigen::VectorXd r;
    Eigen::VectorXi red_idxs;
    Eigen::MatrixXd P0, P, cholQ, R, H;

    STATE ac_state_, newState;
    double TOL;
    int q;
    Eigen::Vector3d g_;

};

} 

#endif 