#include "ukf_imu_uwb_fusion.h"
#include "util.hpp"
#include "ukf_types.h"
#include <vector>
#include <Eigen/Dense>
#include <Eigen/cholesky>
#include <cmath>
#include <iomanip>

namespace UKF
{

ImuUwbFusionUkf::ImuUwbFusionUkf():

 Rot(Eigen::Matrix3d::Identity()), 
  v(Eigen::Vector3d::Zero()), 
  p(Eigen::Vector3d::Zero()), 
  b_gyro(Eigen::Vector3d::Zero()), 
  b_acc(Eigen::Vector3d::Zero()), 
  Q(Eigen::Matrix<double, 12, 12>::Zero()), 
  alpha(Eigen::Vector5d::Constant(1e-3)), 
  red_idxs(Eigen::VectorXi::LinSpaced(15, 0, 14)), 
  P0(Eigen::MatrixXd::Zero(15, 15)), 
  P(P0), 
  cholQ(Q.llt().matrixL().transpose()), 
  TOL(1e-9), 
  q(Q.rows()), 
  g_(Eigen::Vector3d(0.0, 0.0, -9.81)), 
  R(Eigen::Matrix3d::Identity()*0.00002),
  ac_state_(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()), 
  newState(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())
{
    newState.Rot.setZero();
    newState.v.setZero();
    newState.p.setZero();
    newState.b_gyro.setZero();
    newState.b_acc.setZero();
    ac_state_ = newState;
}

ImuUwbFusion::~ImuUwbFusion() {}

Eigen::Matrix3d ImuUwbFusionUkf::_Exp(const Eigen::Vector3d& gyroDt)
{
    double angle = gyroDt.norm(); // imu data로부터 dt를 곱해져셔 처음 값이 들어옴 정규화 시켜서 gyto*dt값의 크기를 angle로 지정
    Matrix3d Rot; // 결과 회전 행렬

    if (angle < TOL) {
        // 각속도가 매우 작을 경우 tangent space에서 거의 크기 변화 없음 , 1차 Taylor 급수 근사 사용
        Rot = Matrix3d::Identity() + wedge(gyroDt); //wegde함수는 gyro*dt값 [angle_x, angle_y, angle_z]의 skew-symmmetric 행렬 생성
    } else {
        // 각속도가 충분히 클 경우, Rodrigues' rotation formula 사용
        Vector3d axis = gyroDt / angle; // 쿼터니언에서 회전 단위 회전축 이므로 크기 1로 맞춰줌
        double c = cos(angle); 
        double s = sin(angle); 

        Rot = c * Matrix3d::Identity() + 
                (1 - c) * axis * axis.transpose() + 
                s * wedge(axis); //로드리게스 식 참고 skew-symmetric matrix도 정규화된 벡터로 부터 나오는 것 주의
    }
    return Rot;
}


Eigen::Vector3d ImuUwbFusionUkf::_Log(const Eigen::Matrix3d& Rot) {
    double cos_angle = 0.5 * Rot.trace() - 0.5;
    cos_angle = std::max(-1.0, std::min(cos_angle, 1.0)); // 값을 -1과 1 사이로 제한
    double angle = std::acos(cos_angle);
    Eigen::Vector3d phi;

    if (std::abs(angle) < TOL) { //angle값이 엄청 작을 경우 tangent space에서 거의 변화 없는경우 
        phi = vee(Rot - Eigen::Matrix3d::Identity()); // R~= R(t0)+R'(t0)(t-t0), t0->0, R->I
                                                      //   = I + Phi(=matrix)
    } else {
        phi = vee((0.5 * angle / std::sin(angle)) * (Rot - Rot.transpose())); // Phi = (angle/2sin(angle)*(R-R.T)).vee
    }
    return phi;
}

Eigen::Vector3d ImuUwbFusionUkf::vee(const Eigen::Matrix3d& Phi) {
    return Eigen::Vector3d(Phi(2, 1), Phi(0, 2), Phi(1, 0)); //skew-symmetric에서 다시 벡터로
}

Eigen::Matrix3d ImuUwbFusionUkf::wedge(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

void ImuUwbFusionUkf::imuInit(const vector<ImuData<double>> &imu_datas)
{
    int num = 0;
    Eigen::Vector3d total_acc(0.0, 0.0, 0.0);
    Eigen::Vector3d total_gyr(0.0, 0.0, 0.0);
    Eigen::Vector3d mean_acc(0.0, 0.0, 0.0);
    Eigen::Vector3d mean_gyr(0.0, 0.0, 0.0);
    for (int i = 0; i < imu_datas.size(); i++)
    {
        total_acc += imu_datas[i].acc;
        total_gyr += imu_datas[i].gyr;
        num++;
    }
    mean_acc = total_acc / num;
    mean_gyr = total_gyr / num;

    newState.b_gyro = mean_gyr;
    newState.Rot = rotationMatrixFromVectors(-g_, mean_acc);
    newState.b_acc = ac_state_.Rot * g_ + mean_acc;
    ac_state_ = newState;
    // cout << "use " << num << " imu datas to init imu pose and bias" << endl;
    // cout << " : " << no_state_.w_b[0] << " " << no_state_.w_b[1] << " " << no_state_.w_b[2] << endl;
    // cout << "init imu bias_a : " << no_state_.a_b[0] << " " << no_state_.a_b[1] << " " << no_state_.a_b[2] << endl;
    // cout << "init imu attitude : " << no_state_.q.w() << " " << no_state_.q.x() << " " << no_state_.q.y() << " " << no_state_.q.z() << endl;
}

void ImuUwbFusion::updateQ(double dt)
{
    Qi_.setIdentity();
    Qi_.block<3, 3>(0, 0) *= sigma_an_2_ * dt * dt;
    Qi_.block<3, 3>(3, 3) *= sigma_wn_2_ * dt * dt;
    Qi_.block<3, 3>(6, 6) *= sigma_aw_2_ * dt;
    Qi_.block<3, 3>(9, 9) *= sigma_ww_2_ * dt;
}

Eigen::Matrix3d rotationMatrixFromVectors(Eigen::Vector3d v0, Eigen::Vector3d v1) {
    v0.normalize(); 
    v1.normalize();

    Eigen::Vector3d axis = v0.cross(v1);

    double angle = acos(v0.dot(v1));

    Eigen::Matrix3d K;
    K <<     0, -axis.z(),  axis.y(),
         axis.z(),      0, -axis.x(),
        -axis.y(),  axis.x(),      0;

    Eigen::Matrix3d rotationMatrix = Eigen::Matrix3d::Identity() + sin(angle) * K + (1 - cos(angle)) * K * K;

    return rotationMatrix;
}

Eigen::VectorXd phi_inv(const STATE& state, const STATE& hat_state) {
    Eigen::Matrix3d dRot = hat_state.Rot * state.Rot.transpose();
    Eigen::Vector3d dRotVec = _Log(dRot);
    Eigen::Vector3d dv = hat_state.v - state.v; 
    Eigen::Vector3d dp = hat_state.p - state.p; 
    Eigen::Vector3d db_gyro = hat_state.b_gyro - state.b_gyro; 
    Eigen::Vector3d db_acc = hat_state.b_acc - state.b_acc;

    Eigen::VectorXd xi(15);
    xi << dRotVec, dv, dp, db_gyro, db_acc;
    return xi;
}

Eigen::Vector3d h(const STATE &State)
{
    Eigen::Vector y;
    y = State.p;
    return y;
}

void ImuUwbFusionUkf::StatePropagation(const vector<ImuData<double>> &last_imu_data, const vector<ImuData<double>> &imu_data) 
{
    double dt = imu_data.stamp - last_imu_data.stamp;
    Vector6d w = Vector6d::Zero();
    STATE newState = f(ac_state_, imu_data, w, dt);
    
}

static STATE f(const STATE& State, const Omega& omega, const Eigen::VectorXd& w, double dt) 
{
    Eigen::Vector3d gyro = omega.gyro - State.b_gyro + w.segment<3>(0);
    Eigen::Vector3d acc = State.Rot * (omega.acc - State.b_acc + w.segment<3>(3)) + g;
    STATE new_state;
    new_state.Rot = State.Rot * _Exp(gyro * dt); 
    new_state.v = State.v + acc * dt;
    new_state.p = State.p + State.v * dt + 0.5 * acc * dt * dt;
    new_state.b_gyro = State.b_gyro;
    new_state.b_acc = State.b_acc;
    return new_state;
}
static STATE getState()
{
    return ac_state_;
}
void ImuUwbFusionUkf::F_num(const vector<ImuData<double>> &imu_data, double dt) 
{
    int d = red_idxs.size();
    MatrixXd P_red = P.block(0, 0, d, d) + TOL * MatrixXd::Identity(d, d);
    VectorXd w = VectorXd::Zero(q);
    MatrixXd xis = weights.sqrt_d_lambda * P_red.llt().matrixL().transpose();
    MatrixXd new_xis = MatrixXd::Zero(2 * d, d);

    for (int j = 0; j < d; ++j) {
        auto s_j_p = red_phi(ac_state_, xis.col(j));
        auto s_j_m = red_phi(ac_state_, -xis.col(j));
        auto new_s_j_p = f(s_j_p, imu_data, w, dt);
        auto new_s_j_m = f(s_j_m, imu_data, w, dt);

        new_xis.row(j) = red_phi_inv(new_state, new_s_j_p).transpose();
        new_xis.row(d + j) = red_phi_inv(new_state, new_s_j_m).transpose();
    }
    VectorXd new_xi = weights.wj * new_xis.colwise().sum();
    new_xis.rowwise() -= new_xi.transpose();
    MatrixXd Xi = weights.wj * new_xis.transpose() * MatrixXd::Identity(d, d).replicate(2, 1);
    F.block(0, 0, d, d) = P_red.llt().solve(Xi.transpose()).transpose();
}

void ImuUwbFusionUkf::G_num(const vector<ImuData<double>> &imu_data, double dt) 
{
    int d = red_idxs.size(); 
    int q = cholQ.rows();

    MatrixXd new_xis = MatrixXd::Zero(2 * q, d);

    for (int j = 0; j < q; ++j) {
        VectorXd w_p = weights.q.sqrt_d_lambda * cholQ.row(j).transpose();
        VectorXd w_m = -weights.q.sqrt_d_lambda * cholQ.row(j).transpose();

        auto new_s_j_p = f(STATE, imu_data, w_p, dt);
        auto new_s_j_m = f(STATE, imu_data, w_m, dt);

        new_xis.row(j) = red_phi_inv(new_state, new_s_j_p).transpose();
        new_xis.row(q + j) = red_phi_inv(new_state, new_s_j_m).transpose();
    }

    VectorXd new_xi = weights.q.wj * new_xis.colwise().sum();
    new_xis.rowwise() -= new_xi.transpose();

    MatrixXd Xi = weights.q.wj * new_xis.transpose() * MatrixXd::Identity(d, d).replicate(2, 1);

    G = MatrixXd::Zero(P.rows(), q);
    for (int i = 0; i < red_idxs.size(); ++i) {
        G.row(red_idxs(i)) = Q.llt().solve(Xi.transpose()).transpose().row(i);
    }
}

void ImuUwbFusionUkf::CovPropagation()
{
    P = F*P*(F.transpose() + G*Q*G.transpose());
    P = (P+P.transpose())/2;
    ac_state_ = newState;

}

void ImuUwbFusionUkf::update(const UwbData<double> &uwb_data, const Eigen::MatrixXd& R)
{
    H_num(uwb_data,up_idx, R);
    state_update();
}

void ImuUwbFusionUkf::H_num(const UwbData<double> &uwb_data, const Eigen::VectorXi& idxs, const Eigen::MatrixXd& __R) 
{
        Eigen::MatrixXd P_red(idxs.size(), idxs.size());
        for (int i = 0; i < idxs.size(); ++i) {
            for (int j = 0; j < idxs.size(); ++j) {
                P_red(i, j) = P(idxs(i), idxs(j));
            }
        }
        
        P_red += TOL * Eigen::MatrixXd::Identity(idxs.size(), idxs.size());

        // Sigma points 계산
        Eigen::MatrixXd xis = weights.up_d.sqrt_d_lambda * P_red.llt().matrixL().transpose();
        Eigen::MatrixXd y_mat = Eigen::MatrixXd::Zero(2 * idxs.size(), y.size());
        Eigen::VectorXd hat_y = h(ac_state_); 
        for (int j = 0; j < idxs.size(); ++j) {
            STATE s_j_p = up_phi(ac_state_, xis.row(j));
            STATE s_j_m = up_phi(ac_state_, -xis.row(j));
            y_mat.row(j) = h(s_j_p);
            y_mat.row(idxs.size() + j) = h(s_j_m);
        }

        Eigen::VectorXd y_bar = weights.up_d.wm * hat_y + weights.up_d.wj * y_mat.colwise().sum();
        y_mat = y_mat.colwise() - y_bar;

        Eigen::MatrixXd Y = weights.up_d.wj * y_mat.transpose() * (xis.rowwise().replicate(2) - xis.rowwise().replicate(2));
        Eigen::MatrixXd H_idx = P_red.llt().solve(Y.transpose()).transpose();

        Eigen::MatrixXd _H = Eigen::MatrixXd::Zero(uwb_data.size(), P.rows());
        for (int i = 0; i < idxs.size(); ++i) {
            _H.col(idxs(i)) = H_idx.col(i);
        }

        Eigen::VectorXd _r = uwb_data - y_bar;

        Eigen::MatrixXd H_new(H.rows() + _H.rows(), H.cols());
        H_new << H,
                _H;
        H = H_new;

        Eigen::MatrixXd r_new(r.rows(), r.cols() + _r.cols());
        r_new << r,_r;
        r = r_new;

        Eigen::MatrixXd R_new(R.rows() + __R.rows(), R.cols() + __R.cols());

        R_new.setZero(); 
        R_new.topLeftCorner(3, 3) = R;
        R_new.bottomRightCorner(3, 3) = __R;
        R = R_new;

}   

void ImuUwbFusionUkf::state_update()
{
    Eigen::MatrixXd S = H*P*H.transpose() + R;
    Eigen::MatrixXd K = S.llt().solve((P*H.transpose()).transpose()).transpose(); 

    Eigen::MatrixXd xi = K*r;
    ac_state_ = red_phi(ac_state_, xi);
    Eigen::Matrix P_new;
    P_new = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;
    P = (P_new + P_new.transpose())/2;
    H.setZero();
    r.setZero();
    R.setZero();
}

void ImuUwbFusion::cfgImuVar(double sigma_an, double sigma_wn, double sigma_aw, double sigma_ww)
{
    sigma_an_2_ = sigma_an;
    sigma_wn_2_ = sigma_wn;
    sigma_aw_2_ = sigma_aw;
    sigma_ww_2_ = sigma_ww;
}


} 
