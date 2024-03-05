#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from sensor_msgs.msg import Imu
from uwb_imu.msg import UwbMsg
import numpy as np
import math

class filtering():
    def __init__(self):
        rospy.init_node("data_generation")
        self.current_state = State()
        self.radius = 5
        self.cnt = 0
        self.rate = rospy.Rate(20)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_cb)
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arming_client = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        rospy.wait_for_service("/mavros/set_mode")
        self.current_yaw = 0.0
        self.set_mode_client = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        self.run()

    def state_cb(self, msg):
        self.current_state = msg
    
    def uwb_callback(self, msg):
        # self.cnt += 1
        # if self.cnt >= 10:
            # noise_level = 0.01
        noise_x = np.random.normal(0, 0.02)
        noise_y = np.random.normal(0, 0.02)
        noise_z = np.random.normal(0, 0.02)

        pub = rospy.Publisher("/uwb",UwbMsg,queue_size=10)

        filter_data = UwbMsg()
        self.header = msg.header
        filter_data.header = msg.header
        filter_data.pos_x = msg.pose.position.x+noise_x
        filter_data.pos_y = msg.pose.position.y+noise_y
        filter_data.pos_z = msg.pose.position.z+noise_z
        pub.publish(filter_data)
            # self.cnt = 0

    def imu_callback(self, msg):
        # self.cnt += 1
        # if self.cnt >= 2:
            # noise_level = 0.01
            noise_x = np.random.normal(0, 0.005)
            noise_y = np.random.normal(0, 0.005)
            noise_z = np.random.normal(0, 0.005)

            pub = rospy.Publisher("/imu/data",Imu,queue_size=10)

            filter_data = Imu()
            
            filter_data.header = msg.header
            filter_data.orientation = msg.orientation
            filter_data.orientation_covariance = msg.orientation_covariance

            filter_data.angular_velocity.x = msg.angular_velocity.x+noise_x
            filter_data.angular_velocity.y = msg.angular_velocity.y+noise_y
            filter_data.angular_velocity.z = msg.angular_velocity.z+noise_z

            filter_data.linear_acceleration.x = msg.linear_acceleration.x+noise_x
            filter_data.linear_acceleration.y = msg.linear_acceleration.y+noise_y
            filter_data.linear_acceleration.z = msg.linear_acceleration.z+noise_z
            # filter_data.linear_acceleration_covariance = msg.linear_acceleration_covariance
            pub.publish(filter_data)
            # self.cnt = 0

    def moving(self):
        pose = PoseStamped()
        pose.pose.position.x = self.radius * math.cos(self.current_yaw)
        pose.pose.position.y = self.radius * math.sin(self.current_yaw)
        pose.pose.position.z = 2.5    
        pose.pose.orientation.x = self.radius *math.cos(self.current_yaw/2+math.pi/4)
        pose.pose.orientation.y = self.radius *math.sin(self.current_yaw/2+math.pi/4)
        return pose
    
    def update_position(self):
        self.current_yaw += 0.01
        pose = PoseStamped()
        if self.current_yaw >= 2 * math.pi:
            self.current_yaw = 0.0
        pose = self.moving()
        return pose


    def run(self):
        offb_set_mode = SetModeRequest()
        offb_set_mode.custom_mode = 'OFFBOARD'
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        last_req = rospy.Time.now()

        while not rospy.is_shutdown():
            if self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                if self.set_mode_client.call(offb_set_mode).mode_sent:
                    rospy.loginfo("OFFBOARD enabled")
                last_req = rospy.Time.now()
            else:
                if not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    if self.arming_client.call(arm_cmd).success:
                        rospy.loginfo("Vehicle armed")
                    last_req = rospy.Time.now()

            pose = self.update_position()
            self.local_pos_pub.publish(pose)
            rospy.Subscriber("/mavros/imu/data",Imu,self.imu_callback)
            rospy.Subscriber("/mavros/local_position/pose",PoseStamped,self.uwb_callback)


            self.rate.sleep()

if __name__ =="__main__":
    filtering()