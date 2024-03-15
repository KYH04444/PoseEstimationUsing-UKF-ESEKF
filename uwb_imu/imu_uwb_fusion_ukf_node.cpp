#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <thread>
#include <mutex>
#include "ukf_imu_uwb_fusion.h"
#include "uwb_imu/UwbMsg.h"

using namespace std;

int imu_freq = 100; //imu frequency is location frequency

nav_msgs::Path path; //result path

vector<sensor_msgs::ImuConstPtr> imu_buffer;
//vector<sensor_msgs::NavSatFixConstPtr> gps_buffer;
vector<uwb_imu::UwbMsg::ConstPtr> uwb_buffer;
mutex imu_mtx, uwb_mtx;

ImuUwbFusionUkf::ImuUwbFusionUkf imu_uwb_fuser; // fuser object
ros::Publisher traj_puber;
ros::Publisher result_puber;
bool initialized = false;
ImuUwbFusionUkf::ImuData<double> last_uwb_imu;     //last interpolated imu data at uwb time
ImuUwbFusionUkf::ImuData<double> last_imu;         //last imu data for predict
ImuUwbFusionUkf::State<double> last_updated_state; //last updated state by uwb ㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁㅁ

void imuCallback(const sensor_msgs::ImuConstPtr &msg)
{
    unique_lock<mutex> lock(imu_mtx);
    imu_buffer.push_back(msg);
}

void uwbCallback(const uwb_imu::UwbMsg::ConstPtr& msg)
{
    unique_lock<mutex> lock(uwb_mtx);
    uwb_buffer.push_back(msg);
}

void interpolateImuData(const sensor_msgs::ImuConstPtr &first_data, const sensor_msgs::ImuConstPtr &second_data, double cur_stamp, sensor_msgs::Imu &inter_data)
{
    // linear_interpolation (https://blog.naver.com/aorigin/220947541918)
    double first_stamp = first_data->header.stamp.toSec();
    double second_stamp = second_data->header.stamp.toSec();
    double scale = (cur_stamp - first_stamp) / (second_stamp - first_stamp);
    inter_data = *first_data;
    // cout << "interpolate start" << endl;
    inter_data.angular_velocity.x = scale * (second_data->angular_velocity.x - first_data->angular_velocity.x) + first_data->angular_velocity.x;
    inter_data.angular_velocity.y = scale * (second_data->angular_velocity.y - first_data->angular_velocity.y) + first_data->angular_velocity.y;
    inter_data.angular_velocity.z = scale * (second_data->angular_velocity.z - first_data->angular_velocity.z) + first_data->angular_velocity.z;
    inter_data.linear_acceleration.x = scale * (second_data->linear_acceleration.x - first_data->linear_acceleration.x) + first_data->linear_acceleration.x;
    inter_data.linear_acceleration.y = scale * (second_data->linear_acceleration.y - first_data->linear_acceleration.y) + first_data->linear_acceleration.y;
    inter_data.linear_acceleration.z = scale * (second_data->linear_acceleration.z - first_data->linear_acceleration.z) + first_data->linear_acceleration.z;
    // cout << "interpolate end" << endl;

}

ImuUwbFusionUkf::ImuData<double> fromImuMsg(const sensor_msgs::Imu &msg)
{
    ImuUwbFusionUkf::ImuData<double> imu_data;
    imu_data.stamp = msg.header.stamp.toSec();
    imu_data.gyr[0] = msg.angular_velocity.x;
    imu_data.gyr[1] = msg.angular_velocity.y;
    imu_data.gyr[2] = msg.angular_velocity.z;
    imu_data.acc[0] = msg.linear_acceleration.x;
    imu_data.acc[1] = msg.linear_acceleration.y;
    imu_data.acc[2] = msg.linear_acceleration.z;

    return move(imu_data);
}

ImuUwbFusionUkf::UwbData<double> fromUwbMsg(const uwb_imu::UwbMsg &msg)
{
    ImuUwbFusionUkf::UwbData<double> uwb_data;
    uwb_data.data[0] = msg.pos_x;
    uwb_data.data[1] = msg.pos_y;
    // uwb_data.data[2] = msg.pos_z;
    uwb_data.data[2] = msg.pos_z;
    uwb_data.cov.setIdentity();
    uwb_data.cov(0, 0) = 0.00002; // need to fix it
    uwb_data.cov(1, 1) = 0.00002;
    uwb_data.cov(2, 2) = 0.00002;
    return move(uwb_data);
}

void pubResult()
{   
        ImuUwbFusionUkf::State<double> result = imu_uwb_fuser.getState();
        geometry_msgs::PoseStamped pose;
        Eigen::Quaterniond q(result.Rot);
        pose.header.frame_id = "world";
        pose.header.stamp = uwb_buffer[0]->header.stamp;
        pose.pose.position.x = result.p[0];
        pose.pose.position.y = result.p[1];
        pose.pose.position.z = result.p[2];
        pose.pose.orientation.w = result.q.w();
        pose.pose.orientation.x = result.q.x();
        pose.pose.orientation.y = result.q.y();
        pose.pose.orientation.z = result.q.z();
        path.poses.push_back(pose);
        path.header = pose.header;
        traj_puber.publish(path);
        result_puber.publish(pose);
}

void processThread()
{
    ros::Rate loop_rate(imu_freq);
    while (ros::ok())
    {
        unique_lock<mutex> imu_lock(imu_mtx);
        unique_lock<mutex> uwb_lock(uwb_mtx);

        // if no data, no need update state
        if (!imu_buffer.size() && !uwb_buffer.size())
        {
            ROS_INFO_THROTTLE(10, "wait for uwb or imu msg ......");
            imu_lock.unlock();
            uwb_lock.unlock();
            loop_rate.sleep();
            continue;
        }

        // init correlative param
        if (!initialized)
        {
            // wait for enough sensor data to init
            if (!imu_buffer.size() || !uwb_buffer.size())
            {
                ROS_INFO_THROTTLE(10, "wait for uwb or imu msg ......");
                imu_lock.unlock();
                uwb_lock.unlock();
                loop_rate.sleep();
                continue;
            }

            // use imu datas at start to initial imu pose
            vector<Fusion::ImuData<double>> imu_datas;
            for (auto &imu_msg : imu_buffer)
            {
                imu_datas.push_back(fromImuMsg(*imu_msg));
            }
            imu_uwb_fuser.imuInit(imu_datas);

            imu_uwb_fuser.cfgRefUwb(uwb_buffer[0]->pos_x, uwb_buffer[0]->pos_y, uwb_buffer[0]->pos_z);

            auto iter = imu_buffer.begin();
            // cout <<"imu.begin = " << *iter << endl;
                cout <<"imu stamp = "<< (*iter)->header.stamp << "uwb stamp = "<< uwb_buffer[0]->header.stamp<< endl ;
            for (; iter != imu_buffer.end(); iter++)
            {
            // {   cout <<"imu.end = " << (*iter)->header.stamp << endl;
                if ((*iter)->header.stamp > uwb_buffer[0]->header.stamp)
                    break;
            }
            // cout <<"pass2" << endl;
            if (imu_buffer.begin() == iter || imu_buffer.end() == iter) //cant find imu data before or after gps data
            {  
                // cout <<"pass3" << endl;
                if (imu_buffer.begin() == iter){
                    uwb_buffer.erase(uwb_buffer.begin()); //no imu data before first gps data, cant interpolate at gps stamp
                // cout <<"pass4" << endl;
                }
                // cout << "pass5" << endl;
                imu_lock.unlock();
                uwb_lock.unlock();
                loop_rate.sleep();
                //cout << "pass6" << endl;
                continue;
            }
            sensor_msgs::Imu inter_imu;
            double cur_stamp = uwb_buffer[0]->header.stamp.toSec();
            // cout << "pass 6.5" << endl;
            interpolateImuData(*(iter - 1), *iter, cur_stamp, inter_imu);
            // cout << "pass7" << endl;
            //record last gps frame time and interpolated imu data
            last_uwb_imu = fromImuMsg(inter_imu);
            last_uwb_imu.stamp = cur_stamp;
            last_imu = last_uwb_imu;
            last_updated_state = imu_uwb_fuser.getState();

            // delete old imu datas and gps data
            imu_buffer.erase(imu_buffer.begin(), iter);
            uwb_buffer.erase(uwb_buffer.begin());

            imu_lock.unlock();
            uwb_lock.unlock();
            loop_rate.sleep();

            initialized = true;
            // cout <<"init" << endl;
            continue;
        }

        for (auto &imu_msg : imu_buffer)
        {
            ImuUwbFusionUkf::ImuData<double> cur_imu = fromImuMsg(*imu_msg);
            if (cur_imu.stamp > last_imu.stamp)
            {   imu_uwb_fuser.SigmaPoints()
                imu_uwb_fuser.motionModel(last_imu, cur_imu);
                imu_uwb_fuser.UT()
                last_imu = cur_imu;
            }
        }

        // use uwb data to update state
        if (uwb_buffer.size() != 0)
        {
            imu_uwb_fuser.recoverState(last_updated_state);

            // collect imu datas during two neighbor gps frames
            vector<ImuUwbFusionUkf::ImuData<double>> imu_datas(0);
            // search first imu data after gps data
            auto iter = imu_buffer.begin();
 
            for (; iter != imu_buffer.end(); iter++)
            {    
                // cout <<"imu stamp = "<< (*iter)->header.stamp << " uwb stamp = "<< uwb_buffer[0]->header.stamp<< endl ;
                if ((*iter)->header.stamp > uwb_buffer[0]->header.stamp)
                    break;
            }
            if (imu_buffer.end() == iter) // no imu data after first gps data, wait for new imu data
            {   
                // cout << "no imu data" << endl;
                imu_lock.unlock();
                uwb_lock.unlock();
                loop_rate.sleep();
                continue;
            }
            assert(imu_buffer.begin() != iter);
            // add last gps_imu data (interpolated data at gps time)
            if (!imu_buffer.empty()) {
                // cout<<"not empty" << endl;
            } 
            else {
                // cout<<"empty" << endl;
            }
            imu_datas.push_back(last_uwb_imu);
            // add imu data between last gps_imu data and current gps_imu data
            for (auto tmp_iter = imu_buffer.begin(); tmp_iter != iter; tmp_iter++)
                imu_datas.push_back(fromImuMsg(*(*tmp_iter)));
            sensor_msgs::Imu inter_imu;
            double cur_stamp = uwb_buffer[0]->header.stamp.toSec();

            interpolateImuData(*(iter - 1), *iter, cur_stamp, inter_imu); //here
            ImuUwbFusionUkf::ImuData<double> cur_uwb_imu = fromImuMsg(inter_imu);

            cur_uwb_imu.stamp = cur_stamp;
            imu_datas.push_back(cur_uwb_imu);
            // generate uwb data
            ImuUwbFusionUkf::UwbData<double> uwb_data = fromUwbMsg(*uwb_buffer[0]);

            // update state (core)
            imu_uwb_fuser.uwbUpdate(uwb_data, imu_datas);

            // update last data
            last_uwb_imu = cur_uwb_imu;
            last_imu = last_uwb_imu;

            uwb_buffer.erase(uwb_buffer.begin());
            imu_buffer.erase(imu_buffer.begin(), iter);
            last_updated_state = imu_uwb_fuser.getState();
        }

        // publish result
        pubResult();

        imu_lock.unlock();
        uwb_lock.unlock();
        loop_rate.sleep();
        continue;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "imu_uwb_fusion_ukf_node");
    ros::NodeHandle nh, ph("~");

    // load imu param and config fuser
    double sigma_an, sigma_wn, sigma_aw, sigma_ww;
    if (!ph.getParam("sigma_an", sigma_an) || !ph.getParam("sigma_wn", sigma_wn) || !ph.getParam("sigma_aw", sigma_aw) || !ph.getParam("sigma_ww", sigma_ww))
    {
        cout << "please config imu param !!!" << endl;
        return 0;
    }
    imu_uwb_fuser.cfgImuVar(sigma_an, sigma_wn, sigma_aw, sigma_ww);

    // load imu freqency param
    if (!ph.getParam("imu_freq", imu_freq))
    {
        cout << "no config imu_freq param, use default 100 !" << endl;
    }

    ros::Subscriber uwb_sub = nh.subscribe<uwb_imu::UwbMsg>("/uwb_filtered", 10, uwbCallback);
    ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>("/imu/data", 10, imuCallback);
    traj_puber = nh.advertise<nav_msgs::Path>("traj", 1);
    result_puber = nh.advertise<geometry_msgs::PoseStamped>("result", 1);

    // start process
    thread process_thread(processThread);

    ros::MultiThreadedSpinner spinner(2);
    spinner.spin();

    process_thread.join();

    return 0;
}
