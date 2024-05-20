import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
class SE3EKF:
    def __init__(self):
        self.Npt = 10000
        self.state = {
            "p": np.zeros(3),  
            "R": np.eye(3),   
            "v": np.zeros(3),  
            "b_acc": np.zeros(3),  
            "b_gyro": np.zeros(3) 
        }
        self.P = np.zeros((15, 15), dtype=np.float32) 
        self.Q = 0.0001 * np.eye(15, dtype=np.float32)  
        self.R = 0.08 * np.eye(5, dtype=np.float32)     
        self.m_vecZ = np.zeros(5, dtype=np.float32)
        self.uwb_position = np.array([
            [4050, -3900, 1200, -900, 4200],
            [2550, 2400, 3000, -3600, -3000],
            [2270, 1290, 430, 670, 1120]
        ], dtype=np.float32)

    def getvecZ(self):
        return self.m_vecZ
    
    def setIMU(self, acc, gyro, delta_t):
        acc = self.state["R"] @(acc - self.state["b_acc"]) + np.array([0, 0, -9.81])
        gyro = gyro - self.state["b_gyro"]
        self.state["R"] = self.state["R"] @ self.exp_map(gyro * delta_t)
        self.state["v"] += acc * delta_t
        self.state["p"] += self.state["v"] * delta_t

    def exp_map(self, omega):
        angle = np.linalg.norm(omega)
        if angle < 1e-9:
            return np.eye(3)
        axis = omega / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.cos(angle) * np.eye(3) + (1 - np.cos(angle))*np.outer(axis , axis) + np.sin(angle) * K 

    def motion_model(self, delta_t):
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:9] = self.state["R"] * delta_t
        self.P = F @ self.P @ F.T + self.Q

    def measurement_model(self):
        h = np.zeros(5)
        for i in range(5):
            h[i] = np.linalg.norm(self.state["p"] - self.uwb_position[:, i])
        return h

    def select_top_4_ranges(self, ranges):
        noise = np.abs(ranges - self.measurement_model())# 측정값의 노이즈 계산 (예를 들어, 절대 값 차이로 계산)
        sorted_indices = np.argsort(noise)
        top_4_indices = sorted_indices[:4] # 일단 4개로 정렬하자 
        
        return ranges[top_4_indices], top_4_indices

    # def correction(self):
    #     H = np.zeros((4, 15))
    #     ranges = self.getvecZ()
    #     selected_ranges, top_4_indices = self.select_top_4_ranges(ranges)
    #     for idx, i in enumerate(top_4_indices):
    #         diff = self.state["p"] - self.uwb_position[:, i]
    #         dist = np.linalg.norm(diff)
    #         H[idx, 0:3] = diff / dist

    #     residual = selected_ranges - self.measurement_model()[top_4_indices]
    #     S = H @ self.P @ H.T + self.R[:4, :4]
    #     K = self.P @ H.T @ np.linalg.inv(S)
    #     dx = K @ residual
    #     self.state["p"] += dx[0:3]
    #     self.state["v"] += dx[3:6]
    #     self.state["R"] = self.state["R"] @ self.exp_map(dx[6:9])
    #     self.state["b_gyro"] += dx[9:12]
    #     self.state["b_acc"] += dx[12:15]
    #     # self.P = (np.eye(15) - K @ H) @ self.P
    #     self.P = (np.eye(15) - K @ H) @ self.P @(np.eye(15) - K @ H).T

    def correction(self):
        H = np.zeros((5, 15))
        for i in range(5):
            diff = self.state["p"] - self.uwb_position[:, i]
            dist = np.linalg.norm(diff)
            H[i, 0:3] = diff / dist
        residual = self.m_vecZ - self.measurement_model()
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ residual
        self.state["p"] += dx[0:3]
        self.state["v"] += dx[3:6]
        self.state["R"] = self.state["R"] @ self.exp_map(dx[6:9])
        self.state["b_gyro"] += dx[9:12]
        self.state["b_acc"] += dx[12:15]
        self.P = (np.eye(15) - K @ H) @ self.P 
        # self.P = (np.eye(15) - K @ H) @ self.P @(np.eye(15) - K @ H).T

    def getState(self):
        return self.state

    def getPmat(self):
        return self.P
    
class EKFNode:
    def __init__(self):
        self.ekf_tightly_coupled = SE3EKF()
        ranges = []
        imu = []
        odom = []
        self.ranges_1 = []
        self.ranges_2 = []
        self.ranges_3 = []
        self.ranges_4 = []
        self.ranges_5 = []

        self.timestamp     = []
        self.linear_acc_x  = []
        self.linear_acc_y  = []
        self.linear_acc_z  = []
        self.angular_vel_x = []
        self.angular_vel_y = []
        self.angular_vel_z = []


        self.gt_x = []
        self.gt_y = []
        self.gt_z = []
        self.esti_x = []
        self.esti_y = []
        self.esti_z = []
        self.time = []

        self.imu_x = []
        self.imu_y = []

        with open('uwb_ranges.txt', 'r') as file:
            ranges = file.readlines()
        with open('odom.txt', 'r') as file:
            odom = file.readlines()

        for line in odom:
            a, b =map(float, line.strip().split("\t"))
            self.gt_x.append(b*-1000)
            self.gt_y.append(a*1000)
            self.gt_z.append(420)

        self.gt_x, self.gt_y = self.loop_closure(self.gt_x,self.gt_y)
        for line in ranges:
            _ ,a, b, c, d, e =map(float, line.strip().split("\t"))
            self.ranges_1.append(a)
            self.ranges_2.append(b)
            self.ranges_3.append(c)
            self.ranges_4.append(d)
            self.ranges_5.append(e)

        with open('los_imu.txt', 'r') as file:
            imu = file.readlines()
        
        for line in imu:
            a, b, c, d, e, f, g = map(float, line.strip().split("\t"))
            self.timestamp.append(a)
            self.linear_acc_x.append(b)
            self.linear_acc_y.append(c)
            self.linear_acc_z.append(d)
            self.angular_vel_x.append(e)
            self.angular_vel_y.append(f)
            self.angular_vel_z.append(g)

        for i in range(len(self.timestamp)):
            
            delta_t = 0.05
            if self.ranges_1[i] == 0:
                self.ranges_1[i] = self.ranges_1[i-1]

            if self.ranges_2[i] == 0:
                self.ranges_2[i] = self.ranges_2[i-1]

            if self.ranges_3[i] == 0:
                self.ranges_3[i] = self.ranges_3[i-1]

            if self.ranges_4[i] == 0:
                self.ranges_4[i] = self.ranges_4[i-1]

            if self.ranges_5[i] == 0:
                self.ranges_5[i] = self.ranges_5[i-1]

            self.ekf_tightly_coupled.getvecZ()[0] = self.ranges_1[i] 
            self.ekf_tightly_coupled.getvecZ()[1] = self.ranges_2[i] 
            self.ekf_tightly_coupled.getvecZ()[2] = self.ranges_3[i]
            self.ekf_tightly_coupled.getvecZ()[3] = self.ranges_4[i]
            self.ekf_tightly_coupled.getvecZ()[4] = self.ranges_5[i]
            acc = np.array([self.linear_acc_x[i], self.linear_acc_y[i],self.linear_acc_z[i]])
            ang = np.array([self.angular_vel_x[i],self.angular_vel_y[i],self.angular_vel_z[i]])
            self.ekf_tightly_coupled.setIMU(acc,ang, delta_t) # self.imu_x.append(self.ekf_tightly_coupled.getImuPosition()[0]) 
            # self.imu_y.append(self.ekf_tightly_coupled.getImuPosition()[1]) 
            self.ekf_tightly_coupled.motion_model(delta_t)
            # if self.ranges_1[i] != 0 or self.ranges_2[i] != 0 or self.ranges_3[i] != 0 or self.ranges_4[i] != 0 or self.ranges_5[i] != 0:  
            self.ekf_tightly_coupled.correction()
            self.time.append(i)

            self.esti_x.append(self.ekf_tightly_coupled.getState()["p"][0])
            self.esti_y.append(self.ekf_tightly_coupled.getState()["p"][1])
            self.esti_z.append(self.ekf_tightly_coupled.getState()["p"][2])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.esti_x, self.esti_y, self.esti_z, linestyle='-', color='red', label='Estimated')
        ax.plot(self.gt_x, self.gt_y, self.gt_z, linestyle='-', color='black', label='Odom')

        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.legend()
        plt.show()

    def loop_closure(self, x, y):
        x_adjusted = np.array(x)
        y_adjusted = np.array(y)

        dx = x[0] - x[-1]
        dy = y[0] - y[-1]

        n_points = len(x)

        for i in range(n_points):
            x_adjusted[i] += dx * (i / n_points)
            y_adjusted[i] += dy * (i / n_points)
        
        return x_adjusted, y_adjusted           


if __name__ =="__main__":
    EKFNode()