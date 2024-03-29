#ifndef _UKF_IMU_UWB_FUSION_H_
#define _UKF_IMU_UWB_FUSION_H_

#include <iostream>
#include <vector>
#include "ukf_types.h"
#include <Eigen/Dense>
// #include <Eigen/Cholesky>
#include <vector>
using namespace std;
using namespace Eigen;
namespace UKF {

class ImuUwbFusionUkf {
public:
    ImuUwbFusionUkf();
    ~ImuUwbFusionUkf();
    void imuInit(const vector<ImuData<double>> &imu_datas);
    void H_num(const UwbData<double> &uwb_data, const Eigen::VectorXd& idxs, const Eigen::MatrixXd& __R);
    void update(const UwbData<double> &uwb_data, const Eigen::MatrixXd& R);
    void F_num(const ImuData<double> &imu_data, double dt);
    void G_num(const ImuData<double> &imu_data, double dt);
    void updateQ(double dt);
    void state_update();
    void cfgImuVar(double sigma_an, double sigma_wn, double sigma_aw, double sigma_ww);
    void cfgRefUwb(double position_x, double position_y, double position_z);
    float StatePropagation(const ImuData<double> &last_imu_data, const ImuData<double> &imu_data);
    void CovPropagation();
    Eigen::Vector3d vee(const Eigen::Matrix3d &Phi);
    Eigen::Matrix3d wedge(const Eigen::Vector3d &v);
    Eigen::Matrix3d Exp(const Eigen::Vector3d &gyroDt);
    Eigen::Vector3d Log(const Eigen::Matrix3d &Rot);
    STATE red_phi(const UKF::STATE &state, const Eigen::VectorXd &xi);
    STATE getState();
    STATE getNewState();
    STATE f(const UKF::STATE& State, const ImuData<double> &imu_data, const Eigen::VectorXd& w, double dt); 
    STATE up_phi(const UKF::STATE &state, const Eigen::VectorXd &xi);
    STATE phi(const UKF::STATE &state, const Eigen::VectorXd xi);
    Eigen::VectorXd phi_inv(const UKF::STATE &state, const UKF::STATE &hat_state);
    Eigen::Matrix3d rotationMatrixFromVectors(Eigen::Vector3d v0, Eigen::Vector3d v1);
    Eigen::Vector3d h(const UKF::STATE &State);

private:
    Eigen::Matrix3d Rot;
    Eigen::Vector3d v, p, b_gyro, b_acc, up_idx, y, g_;
    Eigen::Matrix<double, 12, 12> Q;
    Eigen::VectorXd alpha;
    Eigen::VectorXd r;
    Eigen::VectorXi red_idxs;
    Eigen::MatrixXd P0, F, P, cholQ, R, H, G;
    Eigen::Matrix<double, 12, 12> Qi_;
    double sigma_an_2_;
    double sigma_wn_2_;
    double sigma_aw_2_;
    double sigma_ww_2_;

    // gps reference latitude and longitute
    double ref_position_x_;
    double ref_position_y_;
    double ref_position_z_;
    STATE ac_state_, newState;
    WEIGHTS weights;

    double TOL;
    int q;
};

} 

#endif 
