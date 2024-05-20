import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

class SE3UKF:
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
        self.alpha = 1e-3
        self.kappa = 0
        self.beta = 2
        self.n = 15
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        self.weights_mean = np.zeros(2 * self.n + 1)
        self.weights_cov = np.zeros(2 * self.n + 1)
        self.weights_mean[0] = self.lam / (self.n + self.lam)
        self.weights_cov[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)
        for i in range(1, 2 * self.n + 1):
            self.weights_mean[i] = self.weights_cov[i] = 0.5 / (self.n + self.lam)
    
    def getvecZ(self):
        return self.m_vecZ
    
    def setIMU(self, acc, gyro, delta_t):
        acc = self.state["R"] @ (acc - self.state["b_acc"]) + np.array([0, 0, -9.81])
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
        return np.cos(angle) * np.eye(3) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * K

    def motion_model(self, delta_t):
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:9] = self.state["R"] * delta_t
        self.P = F @ self.P @ F.T + self.Q

    def measurement_model(self, state):
        h = np.zeros(5)
        for i in range(5):
            h[i] = np.linalg.norm(state["p"] - self.uwb_position[:, i])
        return h

    def sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state_to_vector(self.state)
        sqrt_P = np.linalg.cholesky((self.n + self.lam) * self.P + 1e-9 * np.eye(self.n))  # 음수나오면 안돼서 1e-9추가
        for i in range(self.n):
            sigma_points[i + 1] = sigma_points[0] + sqrt_P[i]
            sigma_points[self.n + i + 1] = sigma_points[0] - sqrt_P[i]
        return sigma_points

    def state_to_vector(self, state):
        return np.hstack([state["p"], state["v"], self.log_map(state["R"]), state["b_acc"], state["b_gyro"]])

    def vector_to_state(self, vec):
        state = {
            "p": vec[0:3],
            "v": vec[3:6],
            "R": self.exp_map(vec[6:9]),
            "b_acc": vec[9:12],
            "b_gyro": vec[12:15]
        }
        return state

    def log_map(self, R):
        angle = np.arccos((np.trace(R) - 1) / 2)
        if np.abs(angle) < 1e-9:
            return np.zeros(3)
        return angle / (2 * np.sin(angle)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    def unscented_transform(self, sigma_points, weights_mean, weights_cov, noise_cov):
        n_sigma, dim = sigma_points.shape
        mean = np.zeros(dim)
        for i in range(n_sigma):
            mean += weights_mean[i] * sigma_points[i]
        cov = np.zeros((dim, dim))
        for i in range(n_sigma):
            diff = sigma_points[i] - mean
            cov += weights_cov[i] * np.outer(diff, diff)
        cov += noise_cov
        return mean, cov

    def predict(self, delta_t):
        sigma_points = self.sigma_points()
        for i in range(2 * self.n + 1):
            state = self.vector_to_state(sigma_points[i])
            state["p"] += state["v"] * delta_t
            state["v"] += self.state["R"] @ (state["b_acc"] + np.array([0, 0, -9.81])) * delta_t
            state["R"] = state["R"] @ self.exp_map(state["b_gyro"] * delta_t)
            sigma_points[i] = self.state_to_vector(state)
        mean, cov = self.unscented_transform(sigma_points, self.weights_mean, self.weights_cov, self.Q)
        self.state = self.vector_to_state(mean)
        self.P = cov

    def correct(self):
        sigma_points = self.sigma_points()
        z_sigma_points = np.zeros((2 * self.n + 1, 5))
        for i in range(2 * self.n + 1):
            state = self.vector_to_state(sigma_points[i])
            z_sigma_points[i] = self.measurement_model(state)
        z_mean, z_cov = self.unscented_transform(z_sigma_points, self.weights_mean, self.weights_cov, self.R)
        cross_cov = np.zeros((self.n, 5))
        for i in range(2 * self.n + 1):
            diff_x = sigma_points[i] - self.state_to_vector(self.state)
            diff_z = z_sigma_points[i] - z_mean
            cross_cov += self.weights_cov[i] * np.outer(diff_x, diff_z)
        K = cross_cov @ np.linalg.inv(z_cov)
        self.state = self.vector_to_state(self.state_to_vector(self.state) + K @ (self.m_vecZ - z_mean))
        self.P = self.P - K @ z_cov @ K.T

    def getState(self):
        return self.state
    
class UKFNode:
    def __init__(self):
        self.ukf = SE3UKF()
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
            
            delta_t = 0.07
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

            self.ukf.getvecZ()[0] = self.ranges_1[i] 
            self.ukf.getvecZ()[1] = self.ranges_2[i] 
            self.ukf.getvecZ()[2] = self.ranges_3[i]
            self.ukf.getvecZ()[3] = self.ranges_4[i]
            self.ukf.getvecZ()[4] = self.ranges_5[i]
            acc = np.array([self.linear_acc_x[i], self.linear_acc_y[i],self.linear_acc_z[i]])
            ang = np.array([self.angular_vel_x[i],self.angular_vel_y[i],self.angular_vel_z[i]])
            self.ukf.setIMU(acc,ang, delta_t)
            self.ukf.predict(delta_t)
            # if self.ranges_1[i] != 0 or self.ranges_2[i] != 0 or self.ranges_3[i] != 0 or self.ranges_4[i] != 0 or self.ranges_5[i] != 0: 
            self.ukf.correct()
            self.time.append(i)

            self.esti_x.append(self.ukf.getState()["p"][0])
            self.esti_y.append(self.ukf.getState()["p"][1])
            self.esti_z.append(self.ukf.getState()["p"][2])

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
    UKFNode()
