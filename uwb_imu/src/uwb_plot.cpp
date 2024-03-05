#include <ros/ros.h>
#include <iostream>
#include <vector>

#include "imu_uwb_fusion.h"
#include "uwb_imu/UwbMsg.h"


class uwbPlot
{
public:
    uwbPlot()
    {
        // 퍼블리쉬 할 토픽 선언
        pub_ = n_.advertise<uwb_imu::UwbMsg>("/uwb_plot", 1000);

        // 서브스크라이브 할 토픽 선언
        sub_ = n_.subscribe("/uwb", 1, &uwbPlot::callback, this);
    }

    void callback(const uwb_imu::UwbMsg& input)
    {
        uwb_imu::UwbMsg output;
        output.pos_x = input.pos_x;
        output.pos_y = input.pos_y;
        output.pos_z = input.pos_z;

        pub_.publish(output);
    }

private:
    ros::NodeHandle n_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "uwb_plot");
    uwbPlot uwbPlotNode;
    ros::spin();
    return 0;
}