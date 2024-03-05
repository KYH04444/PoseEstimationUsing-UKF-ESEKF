#include <ros/ros.h>
#include <iostream>
#include <geometry_msgs/PoseStamped.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <sensor_msgs/Imu.h>
// #include <uwb_imu/UwbMsg.h>
#include <random>
#include "imu_uwb_fusion.h"
#include "uwb_imu/UwbMsg.h"
#include <cmath>
using namespace std;
class Filtering {
public:
    Filtering() : radius(5), cnt(0) {
        ros::NodeHandle nh;
        state_sub = nh.subscribe("/mavros/state", 10, &Filtering::stateCallback, this);
        local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);
        
        arming_client = nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
        set_mode_client = nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
        pub_imu = nh.advertise<sensor_msgs::Imu>("/imu/data", 10);
        pub_uwb = nh.advertise<uwb_imu::UwbMsg>("/uwb", 10);
        uwb_sub = nh.subscribe("/mavros/local_position/pose", 10, &Filtering::uwbCallback, this);
        imu_sub = nh.subscribe("/mavros/imu/data", 10, &Filtering::imuCallback, this);
        while (ros::ok() && !current_state.connected) {
            ros::spinOnce();
            // cout << "i" << endl;
            rate.sleep();

        }

        run();
    }

    void stateCallback(const mavros_msgs::State::ConstPtr& msg) {
        current_state = *msg;
    }

    void uwbCallback(const geometry_msgs::PoseStamped &msg) {
        // cout << "uwb" << endl;
        uwb_imu::UwbMsg filtered_msg;
        filtered_msg.header = msg.header;
        filtered_msg.pos_x = msg.pose.position.x + generateNoise(0, 0.02);
        filtered_msg.pos_y = msg.pose.position.y + generateNoise(0, 0.02);
        filtered_msg.pos_z = msg.pose.position.z + generateNoise(0, 0.02);

        pub_uwb.publish(filtered_msg);
    }

    void imuCallback(const sensor_msgs::Imu &msg) {


        // imu_msg = *msg;
        // cout << "imu" << endl;
        sensor_msgs::Imu filtered_msg;
        filtered_msg.header = msg.header;
        filtered_msg.angular_velocity.x = msg.angular_velocity.x + generateNoise(0, 0.005);
        filtered_msg.angular_velocity.y = msg.angular_velocity.y + generateNoise(0, 0.005);
        filtered_msg.angular_velocity.z = msg.angular_velocity.z + generateNoise(0, 0.005);
        filtered_msg.linear_acceleration.x = msg.linear_acceleration.x + generateNoise(0, 0.005);
        filtered_msg.linear_acceleration.y = msg.linear_acceleration.y + generateNoise(0, 0.005);
        filtered_msg.linear_acceleration.z = msg.linear_acceleration.z + generateNoise(0, 0.005);

        pub_imu.publish(filtered_msg);

    }

    void run() {
        ros::Time last_request = ros::Time::now();

        while (ros::ok()) {
            if (current_state.mode != "OFFBOARD" && (ros::Time::now() - last_request) > ros::Duration(5.0)) {
                mavros_msgs::SetMode offb_set_mode;
                offb_set_mode.request.custom_mode = "OFFBOARD";
                if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent) {
                    ROS_INFO("OFFBOARD enabled");
                }
                last_request = ros::Time::now();
            } else {
                if (!current_state.armed && (ros::Time::now() - last_request) > ros::Duration(5.0)) {
                    mavros_msgs::CommandBool arm_cmd;
                    arm_cmd.request.value = true;
                    if (arming_client.call(arm_cmd) && arm_cmd.response.success) {
                        ROS_INFO("Vehicle armed");
                    }
                    last_request = ros::Time::now();
                }
            }

            geometry_msgs::PoseStamped pose = updatePosition();
            local_pos_pub.publish(pose);

            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber state_sub;
    ros::Publisher local_pos_pub;
    ros::Publisher pub_imu;
    ros::Publisher pub_uwb;
    ros::ServiceClient arming_client;
    ros::Subscriber uwb_sub;
    ros::Subscriber imu_sub;
    ros::ServiceClient set_mode_client;
    mavros_msgs::State current_state;
    uwb_imu::UwbMsg uwb_msg;
    sensor_msgs::Imu imu_msg;
    double radius;
    int cnt;
    double current_yaw = 0;
    ros::Rate rate{20};

    // void publishUWBData() {
    //     uwb_imu::UwbMsg filtered_msg;
    //     filtered_msg.header = uwb_msg.header;
    //     filtered_msg.pos_x = uwb_msg.pos_x + generateNoise(0, 0.02);
    //     filtered_msg.pos_y = uwb_msg.pos_y + generateNoise(0, 0.02);
    //     filtered_msg.pos_z = uwb_msg.pos_z + generateNoise(0, 0.02);
    //     ros::Publisher pub = nh.advertise<uwb_imu::UwbMsg>("/uwb_filtered", 10);
    //     pub.publish(filtered_msg);
    // }

    // void publishIMUData() {
    //     sensor_msgs::Imu filtered_msg = imu_msg;
    //     filtered_msg.angular_velocity.x += generateNoise(0, 0.005);
    //     filtered_msg.angular_velocity.y += generateNoise(0, 0.005);
    //     filtered_msg.angular_velocity.z += generateNoise(0, 0.005);
    //     filtered_msg.linear_acceleration.x += generateNoise(0, 0.005);
    //     filtered_msg.linear_acceleration.y += generateNoise(0, 0.005);
    //     filtered_msg.linear_acceleration.z += generateNoise(0, 0.005);
    //     ros::Publisher pub = nh.advertise<sensor_msgs::Imu>("/imu/data", 10);
    //     pub.publish(filtered_msg);
    // }

    double generateNoise(double mean, double stddev) {
        static std::default_random_engine generator;
        std::normal_distribution<double> distribution(mean, stddev);
        return distribution(generator);
    }

    geometry_msgs::PoseStamped updatePosition() {
        geometry_msgs::PoseStamped pose;
        pose.pose.position.x = radius * std::cos(current_yaw);
        pose.pose.position.y = radius * std::sin(current_yaw);
        pose.pose.position.z = 2.5;
        pose.pose.orientation.x = radius * std::cos(current_yaw/2+M_PI/4);
        pose.pose.orientation.y = radius * std::sin(current_yaw/2+M_PI/4);
        current_yaw += 0.01;
        if (current_yaw >= 2 * M_PI) {
            current_yaw = 0.0;
        }
        return pose;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_generation_cpp");
    Filtering filter;
    return 0;
}
