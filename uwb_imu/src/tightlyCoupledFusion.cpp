#include <Eigen/Dense>
#include <ros/ros.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PointStamped.h>

using namespace Eigen;
using namespace std;

typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;
typedef Eigen::Matrix<double, 2, 5> Matrix2x5d;
typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 2, 2> Matrix2d;
typedef Eigen::Matrix<double, 3, 5> Matrix3x5d;

class TightlyCoupledUwbImu
{
private:
    ros::NodeHandle nh_;
    ros::Publisher state_pub_;

    Vector5d state_;
    Matrix5d P_, Q_;
    Matrix5d R_;
    double dt_;
    Matrix2d Rotation;
    Matrix3x5d uwb_position;

public:
    TightlyCoupledUwbImu() : state_(Vector5d::Zero()), P_(Matrix5d::Identity()), Q_(Matrix5d::Identity()*0.1), R_(Matrix5d::Identity()*0.2), dt_(0.1) {
        state_pub_ = nh_.advertise<geometry_msgs::TwistStamped>("estimated_state", 1);
    void setUwbPosition(){
        uwb_position <<0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0;
    }
    }

    float predict(const ImuData<double> &last_imu_data, const ImuData<double> &imu_data) {
        dt_ = imu_data.stamp - last_imu_data.stamp;
        Matrix5d F = Matrix5d::Identity();
        acc = imu_data.acc;
        omega = imu_data.gyr;
        double angle =omega*dt_;
        Rotation = cos(angle)*Matrix2d::Identity();
        Rotation(0,1) = -sin(angle);
        Rotation(1,0) = sin(angle);
        F(0, 2) = dt_;
        F(1, 3) = dt_;
        F(4, 4) = 1;

        Matrix<double, 5, 2> B;
        B << 0.5 * dt_ * dt_, 0,
             0, 0.5 * dt_ * dt_,
             dt_, 0,
             0, dt_,
             0, 0;

        Vector2d u(acc(0), acc(1));
        Vector5d control = B * Rotation*u;

        state_ = F * state_ + control;
        state_(4) += omega(2) * dt_;  

        P_ = F * P_ * F.transpose() + Q_;
        return dt_
    }

    void update(const UwbData<double> &uwb_data) {
        double px = state_(0), py = state_(1);
        Vector5d expected_z; 
        Matrix5d H;
        for(int i= 0; i<uwb_position.cols(); i++)
        {
            expected_z << sqrt((px - uwb_position(0,i))*(px - uwb_position(0,i)) + (py - uwb_position(1,i))*(py - uwb_position(1,i)));
            H << (px - uwb_position(0,i)) / expected_z(i), (py - uwb_position(1,i)) / expected_z, 0, 0, 0;
        }

        Matrix<double, 5, 1> K = P_ * H.transpose() * (H * P_ * H.transpose() + R_).inverse();

        double y = uwb_data - expected_z; 
        state_ += K * y;

        P_ = (Matrix5d::Identity() - K * H) * P_;

        geometry_msgs::TwistStamped msg;
        msg.header.stamp = ros::Time::now();
        msg.twist.linear.x = state_(0);
        msg.twist.linear.y = state_(1);
        state_pub_.publish(msg);
    }

    // void ImuUwbFusionUkf::imuInit(const vector<ImuData<double>> &imu_datas)
    // {
    //     int num = 0;
    //     Eigen::Vector3d total_acc(0.0, 0.0, 0.0);
    //     Eigen::Vector3d total_gyr(0.0, 0.0, 0.0);
    //     Eigen::Vector3d mean_acc(0.0, 0.0, 0.0);
    //     Eigen::Vector3d mean_gyr(0.0, 0.0, 0.0);
    //     for (int i = 0; i < imu_datas.size(); i++)
    //     {
    //         total_acc += imu_datas[i].acc;
    //         total_gyr += imu_datas[i].gyr;
    //         num++;
    //     }
    //     mean_acc = total_acc / num;
    //     mean_gyr = total_gyr / num;

    //     newState.b_gyro = mean_gyr;
    //     newState.Rot = Eigen::Matrix3d::Zero();
    //     newState.b_acc = ac_state_.Rot * g_ + mean_acc;
    //     ac_state_ = newState;
    // }
    
    void ImuUwbFusionUkf::cfgRefUwb(double position_x, double position_y, double position_z)
    {
        ref_position_x_ = position_x;
        ref_position_y_ = position_y;
        ref_position_z_ = position_z; 
    }
    
    void ImuUwbFusionUkf::cfgImuVar(double sigma_an, double sigma_wn, double sigma_aw, double sigma_ww)
    {
        sigma_an_2_ = sigma_an;
        sigma_wn_2_ = sigma_wn;
        sigma_aw_2_ = sigma_aw;
        sigma_ww_2_ = sigma_ww;
    }
};

// int main(int argc, char **argv)
// {
//     ros::init(argc, argv, "tightly_coupled_uwb_imu");
//     TightlyCoupledUwbImu filter;
//     // filter.setUwbPosition(); // 생성자에 추가해야되는거 생각
//     filter.predict(last_imu,cur_imu);
//     filter.update(uwb_data)
//     //object 생성예정 아니면 따로 node생성해도 되고
//     ros::spin();
//     return 0;
// }
