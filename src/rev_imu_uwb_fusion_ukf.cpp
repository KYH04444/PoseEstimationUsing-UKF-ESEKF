#include "ukf_imu_uwb_fusion.h"
#include <vector>
// #include <Eigen/Dense>
// #include <cmath>
#include <iomanip>
using namespace Eigen;
namespace UKF {

ImuUwbFusionUkf::ImuUwbFusionUkf() :
    Rot(Eigen::Matrix3d::Identity()),
    v(Eigen::Vector3d::Zero()),
    p(Eigen::Vector3d::Zero()),
    b_gyro(Eigen::Vector3d::Zero()),
    b_acc(Eigen::Vector3d::Zero()),
    up_idx(Eigen::Vector3d(6,7,8)),
    y(Eigen::Vector3d::Zero()),
    g_(Eigen::Vector3d(0.0, 0.0, -9.81)),
    alpha(Eigen::VectorXd::Constant(5, 1e-3)),
    Q(Eigen::Matrix<double, 12, 12>::Zero()),
    F(Eigen::Matrix<double, 15, 15>::Zero()),
    cholQ(Eigen::Matrix<double, 12, 12>::Zero()),
    // cholQ(Q.llt().matrixL().transpose()),
    red_idxs(Eigen::VectorXd::LinSpaced(15, 0, 14)),
    P0(Eigen::Matrix<double, 15, 15>::Zero()),
    P(P0),
    H(Eigen::MatrixXd::Zero(0,0)),
    G(Eigen::MatrixXd::Zero(15,12)),
    TOL(1e-9), 
    q(Q.rows()), 
    dt(0),
    r(Eigen::VectorXd::Zero(0)),
    // gnssR(Eigen::Matrix3d::Identity()*0.00002),
    R(Eigen::MatrixXd::Zero(0,0)),
    weights(red_idxs.size(), Q.rows(), up_idx.size(), 0, 0, alpha.data())
{
    newState.Rot.setZero();
    newState.v.setZero();
    newState.p.setZero();
    newState.b_gyro.setZero();
    newState.b_acc.setZero();
    ac_state_ = newState;
}

ImuUwbFusionUkf::~ImuUwbFusionUkf() {}

Eigen::Vector3d ImuUwbFusionUkf::vee(const Eigen::Matrix3d& Phi) 
{
    return Eigen::Vector3d(Phi(2, 1), Phi(0, 2), Phi(1, 0)); //skew-symmetric에서 다시 벡터로
}

Eigen::Vector3d ImuUwbFusionUkf::Log(const Eigen::Matrix3d& Rot) 
{
    double TOL = 1e-9;
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

Eigen::Matrix3d ImuUwbFusionUkf::wedge(const Eigen::Vector3d& v) 
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(),
         v.z(), 0, -v.x(),
        -v.y(), v.x(), 0;
    return m;
}

Eigen::Matrix3d ImuUwbFusionUkf::Exp(const Eigen::Vector3d& gyroDt)
{   
    double TOL = 1e-9;
    double angle = gyroDt.norm(); // imu data로부터 dt를 곱해져셔 처음 값이 들어옴 정규화 시켜서 gyto*dt값의 크기를 angle로 지정
    Matrix3d Rot; 

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

Eigen::VectorXd ImuUwbFusionUkf::phi_inv(const UKF::STATE &state, const UKF::STATE &hat_state) {
    Eigen::Matrix3d dRot = hat_state.Rot * state.Rot.transpose();
    Eigen::Vector3d dRotVec = Log(dRot);
    Eigen::Vector3d dv = hat_state.v - state.v; 
    Eigen::Vector3d dp = hat_state.p - state.p; 
    Eigen::Vector3d db_gyro = hat_state.b_gyro - state.b_gyro; 
    Eigen::Vector3d db_acc = hat_state.b_acc - state.b_acc;
    Eigen::VectorXd xi(15);
    xi << dRotVec, dv, dp, db_gyro, db_acc;
    return xi.transpose();
}

double ImuUwbFusionUkf::getDt()
{
    return dt;
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
    newState.Rot = Eigen::Matrix3d::Zero();
    newState.b_acc = ac_state_.Rot * g_ + mean_acc;
    ac_state_ = newState;
}

void ImuUwbFusionUkf::updateQ()
{
    // Q.setIdentity();
    double imu_std[4] = {sigma_an_2_, sigma_wn_2_, sigma_aw_2_, sigma_ww_2_};
    // Q.block<3, 3>(0, 0) = sigma_an_2_**2 
    // Q.block<3, 3>(3, 3) = sigma_wn_2_**2 
    // Q.block<3, 3>(6, 6) = sigma_aw_2_**2 
    // Q.block<3, 3>(9, 9) = sigma_ww_2_**2 
    // cholQ = Q.llt().matrixL().transpose();

    for(int i = 0; i < 4; ++i) 
    {
        Q.block<3,3>(i*3, i*3) = (imu_std[i] * imu_std[i]) * Eigen::MatrixXd::Identity(3, 3);
    }
    cholQ = Q.llt().matrixL().transpose();
    
}

STATE ImuUwbFusionUkf::phi(const UKF::STATE &state, const Eigen::VectorXd &xi)
{   
    Eigen::Matrix3d Rot = Exp(xi.segment<3>(0))*state.Rot;
    STATE new_state;
    new_state.Rot = Rot; 
    new_state.v = state.v + xi.segment<3>(3);
    new_state.p =  state.p + xi.segment<3>(6);
    new_state.b_gyro = state.b_gyro + xi.segment<3>(9);
    new_state.b_acc = state.b_acc + xi.segment<3>(12);
    return new_state;
}

STATE ImuUwbFusionUkf::up_phi(const UKF::STATE &state, const Eigen::VectorXd &xi)//////////////////////////
{
    STATE new_state;
    new_state.Rot = state.Rot; 
    new_state.v = state.v; 
    new_state.p = xi + state.p; 
    new_state.b_gyro = state.b_gyro;
    new_state.b_acc = state.b_acc;
    return new_state;
}

Eigen::Vector3d ImuUwbFusionUkf::h(const UKF::STATE &State)
{
    Eigen::Vector3d y;
    y = State.p;
    return y;
}

float ImuUwbFusionUkf::StatePropagation(const ImuData<double> &last_imu_data, const ImuData<double> &imu_data) 
{
    dt = imu_data.stamp - last_imu_data.stamp;
    Eigen::VectorXd w;
    w = Eigen::VectorXd::Constant(6,0);
    newState = f(ac_state_, imu_data, w, dt);
    return dt;
}
  
STATE ImuUwbFusionUkf::f(const UKF::STATE &State, const ImuData<double> &imu_data, const Eigen::VectorXd& w, double dt) //////////////////////////
{   Eigen::Vector3d g_; 
    g_ = Eigen::Vector3d(0,0,-9.8);
    Eigen::Vector3d gyro = imu_data.gyr - State.b_gyro + w.segment<3>(0);
    Eigen::Vector3d acc = State.Rot * (imu_data.acc - State.b_acc + w.segment<3>(3)) + g_;
    STATE new_state;
    new_state.Rot = State.Rot * Exp(gyro * dt);
    new_state.v = State.v + acc * dt;
    new_state.p = State.p + State.v * dt + 0.5 * acc * dt * dt;
    new_state.b_gyro = State.b_gyro;
    new_state.b_acc = State.b_acc;
    return new_state;
}

STATE ImuUwbFusionUkf::getState()
{
    return ac_state_;
}

STATE ImuUwbFusionUkf::getNewState()
{
    return newState;
}


void ImuUwbFusionUkf::F_num(const ImuData<double> &imu_data, double dt) 
{
    int d = red_idxs.size();
    MatrixXd P_red = P.block(0, 0, d, d) + TOL * MatrixXd::Identity(d, d); //0,0에서 시작해서 15x15까지 행렬사용
    Eigen::VectorXd w = VectorXd::Zero(q);
    Eigen::MatrixXd matrixL_transposed = P_red.llt().matrixL().transpose(); //p_red의 cholesky분해하고 하삼각행렬의 전치
    Eigen::MatrixXd xis = weights.red_d.sqrt_d_lambda * matrixL_transposed;
    MatrixXd new_xis = MatrixXd::Zero(2 * d, d);
    // STATE ac_state_ = getState();
    // STATE new_state = getNewState();
    for (int j = 0; j < d; ++j) {
        auto s_j_p = phi(ac_state_, xis.row(j));
        auto s_j_m = phi(ac_state_, -xis.row(j));
        auto new_s_j_p = f(s_j_p, imu_data, w, dt);
        auto new_s_j_m = f(s_j_m, imu_data, w, dt);

        new_xis.row(j) = phi_inv(newState, new_s_j_p).transpose();
        new_xis.row(d + j) = phi_inv(newState, new_s_j_m).transpose();
    }

    Eigen::VectorXd new_xi = weights.red_d.wj * new_xis.colwise().sum();
    // std::cout <<"new_xis" << new_xi <<std::endl;
    for (int i = 0; i < new_xis.rows(); ++i) 
    {
        new_xis.row(i) -= new_xi;
    }
    Eigen::MatrixXd stacked_xis(xis.rows() * 2, xis.cols());
    stacked_xis << xis,
                  -xis;
    MatrixXd Xi = weights.red_d.wj * new_xis.transpose() *stacked_xis;
    F.block(0, 0, d, d) = P_red.fullPivLu().solve(Xi.transpose()).transpose();
}

void ImuUwbFusionUkf::G_num(const ImuData<double> &imu_data, double dt) 
{   
    updateQ();
    int d = red_idxs.size(); 
    int q = Q.rows();
    MatrixXd new_xis = MatrixXd::Zero(2 * q, d);
    // std::cout << "cholQ" << cholQ << std::endl;
    for (int j = 0; j < q; ++j) 
    {   
        // STATE ac_state_ = getState();
        // STATE new_state = getNewState();
        Eigen::VectorXd w_p = weights.q.sqrt_d_lambda * cholQ.row(j);
        Eigen::VectorXd w_m = -weights.q.sqrt_d_lambda * cholQ.row(j);

        auto new_s_j_p = f(ac_state_, imu_data, w_p, dt);
        auto new_s_j_m = f(ac_state_, imu_data, w_m, dt);

        new_xis.row(j) = phi_inv(newState, new_s_j_p).transpose();
        new_xis.row(q + j) = phi_inv(newState, new_s_j_m).transpose();
    }

    Eigen::VectorXd new_xi = weights.q.wj * new_xis.colwise().sum();

    for (int i = 0; i < new_xis.rows(); ++i) 
    {
        new_xis.row(i) -= new_xi;
    }
    Eigen::MatrixXd stacked_xis(q * 2, Q.cols());
    stacked_xis << cholQ,
                  -cholQ;
    MatrixXd Xi = weights.q.wj * new_xis.transpose() * stacked_xis;
    // Eigen::MatrixXd G;
    // G = MatrixXd::Zero(P.rows(), q);
    for (int i = 0; i < red_idxs.size(); ++i) {
        G.row(red_idxs(i)) = Q.fullPivLu().solve(Xi.transpose()).transpose().row(i);
    }
}

void ImuUwbFusionUkf::CovPropagation()
{   
    // std::cout <<"FPF" << F*P*F.transpose() <<std::endl;
    // std::cout <<"GQG" << G*Q*G.transpose() <<std::endl;
    P = F*P*F.transpose() + G*Q*G.transpose();
    P = (P+P.transpose())/2;
    ac_state_ = newState;

}

void ImuUwbFusionUkf::update(const UwbData<double> &uwb_data, const Eigen::MatrixXd& R)
{   

    H_num(uwb_data, up_idx, R);
    state_update();
}
void ImuUwbFusionUkf::recoverState(const UKF::STATE &last_updated_state)
{
    ac_state_ = last_updated_state;
}

void ImuUwbFusionUkf::H_num(const UwbData<double> &uwb_data, const Eigen::VectorXd& idxs, const Eigen::MatrixXd& __R) 
{
        Eigen::MatrixXd P_red(idxs.size(), idxs.size());
        for (int i = 0; i < idxs.size(); ++i) {
            for (int j = 0; j < idxs.size(); ++j) {
                P_red(i, j) = P(idxs(i), idxs(j));
            }
        }
        
        P_red = P_red + TOL * Eigen::MatrixXd::Identity(idxs.size(), idxs.size());

        Eigen::MatrixXd matrixL_transposed = P_red.llt().matrixL().transpose();
        Eigen::MatrixXd xis = weights.up_d.sqrt_d_lambda * matrixL_transposed;
        Eigen::MatrixXd y_mat = Eigen::MatrixXd::Zero(2 * P_red.rows(), y.rows());
        Eigen::VectorXd hat_y = h(ac_state_); 
        for (int j = 0; j < idxs.size(); ++j) {
            STATE s_j_p = up_phi(ac_state_, xis.row(j));
            STATE s_j_m = up_phi(ac_state_, -xis.row(j));
            y_mat.row(j) = h(s_j_p);
            y_mat.row(P_red.rows() + j) = h(s_j_m);
        }
        Eigen::VectorXd y_mat_rowsum = weights.up_d.wj * y_mat.colwise().sum();
        Eigen::VectorXd y_bar = weights.up_d.wm * hat_y ;
        // std::cout << "y_mat_rowsum" <<y_mat_rowsum <<std::endl;
        for (int i = 0; i<y_bar.size(); ++i)
        {
            y_bar(i) += y_mat_rowsum(i);
            
        }
        // std::cout <<"y_bar" <<y_bar << std::endl;
        // y_mat = y_mat - y_bar;
        
        for (int i = 0; i < y_mat.rows(); ++i) 
        {
            y_mat.row(i) -= y_bar;
        }
        Eigen::MatrixXd stacked_xis(xis.rows() * 2, xis.cols());
        stacked_xis << xis,
                      -xis;

        Eigen::MatrixXd Y = weights.up_d.wj * y_mat.transpose() * stacked_xis;
        Eigen::MatrixXd H_idx = P_red.fullPivLu().solve(Y.transpose()).transpose();

        Eigen::MatrixXd _H = Eigen::MatrixXd::Zero(uwb_data.data.rows(), P.rows());
        for (int i = 0; i < idxs.size(); ++i) {
            _H.col(idxs(i)) = H_idx.col(i);
        }

        // Eigen::VectorXd _r = uwb_data.data - y_bar;
        VectorXd _r = Eigen::Vector3d::Zero();
        for (int i = 0; i<_r.size(); ++i)
        {   
            _r(i) = uwb_data.data(i) -y_bar(i);
            // y_bar(i) += y_mat_rowsum(i);
            
        }
        // std::cout << "H_" << _H <<std::endl;
        // std::cout << "H" << H <<std::endl;
        // H.resize(_H.rows(), _H.cols());
        // H.setZero();
        // Eigen::MatrixXd H_new(_H.rows() + _H.rows(), H.cols());
        // H_new << H,
        //         _H;
        H = _H;
        r = _r; 
        R = __R;

}   

void ImuUwbFusionUkf::state_update()
{
    Eigen::MatrixXd S = H*P*H.transpose() + R;
    Eigen::MatrixXd K = S.fullPivLu().solve((P*H.transpose()).transpose()).transpose(); 

    Eigen::VectorXd xi = K*r;
    ac_state_ = phi(ac_state_, xi);
     
    Eigen::MatrixXd P_new = (Eigen::MatrixXd::Identity(P.rows(),P.rows()) - K * H) * P;
    P = (P_new + P_new.transpose())/2;
    H.setZero();
    r.setZero();
    R.setZero();
}

void ImuUwbFusionUkf::cfgImuVar(double sigma_an, double sigma_wn, double sigma_aw, double sigma_ww)
{
    sigma_an_2_ = sigma_an;
    sigma_wn_2_ = sigma_wn;
    sigma_aw_2_ = sigma_aw;
    sigma_ww_2_ = sigma_ww;
}

void ImuUwbFusionUkf::cfgRefUwb(double position_x, double position_y, double position_z)
{
    ref_position_x_ = position_x;
    ref_position_y_ = position_y;
    ref_position_z_ = position_z; 
}

} 
