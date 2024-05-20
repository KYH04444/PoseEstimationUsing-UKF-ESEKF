import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D

class SE3ESEKF:
    def __init__(self):
        self.state = {
            "p": np.zeros(3),
            "R": np.eye(3),
            "v": np.zeros(3),
            "a_b": np.zeros(3),
            "w_b": np.zeros(3)
        }
        self.P = np.zeros((15, 15), dtype=np.float32)
        self.Q = np.zeros((15, 15), dtype=np.float32)
        self.V = np.zeros((3, 3), dtype=np.float32)
        self.Fi = np.zeros((15, 15), dtype=np.float32)
        self.Fi[3:15, 0:12] = np.eye(12)
        self.Hx = np.zeros((3, 15), dtype=np.float32)
        self.Hx[0:3, 0:3] = np.eye(3)
        self.g = np.array([0.0, 0.0, -9.81])
        self.ref_position = np.zeros(3)

    def imuInit(self, imu_datas):
        num = len(imu_datas)
        total_acc = np.zeros(3)
        total_gyr = np.zeros(3)
        for data in imu_datas:
            total_acc += data['acc']
            total_gyr += data['gyr']
        mean_acc = total_acc / num
        mean_gyr = total_gyr / num

        self.state['w_b'] = mean_gyr
        self.state['R'] = R.from_rotvec(np.cross(-self.g, mean_acc)).as_matrix()
        self.state['a_b'] = self.state['R'] @ self.g + mean_acc

    def cfgImuVar(self, sigma_an, sigma_wn, sigma_aw, sigma_ww):
        self.Q[0:3, 0:3] = sigma_an * np.eye(3)
        self.Q[3:6, 3:6] = sigma_wn * np.eye(3)
        self.Q[6:9, 6:9] = sigma_aw * np.eye(3)
        self.Q[9:12, 9:12] = sigma_ww * np.eye(3)

    def cfgRefUwb(self, position_x, position_y, position_z):
        self.ref_position = np.array([position_x, position_y, position_z])

    def updateNominalState(self, imu_data, delta_t):
        imu_acc = imu_data['acc']
        imu_gyr = imu_data['gyr']
        R_mat = self.state['R']

        self.state['p'] += self.state['v'] * delta_t + 0.5 * (R_mat @ (imu_acc - self.state['a_b']) + self.g) * delta_t**2
        self.state['v'] += (R_mat @ (imu_acc - self.state['a_b']) + self.g) * delta_t
        q_v = (imu_gyr - self.state['w_b']) * delta_t
        self.state['R'] = self.state['R'] @ R.from_rotvec(q_v).as_matrix()

    def calcF(self, imu_data, delta_t):
        self.Fx = np.eye(15)
        R_mat = self.state['R']
        self.Fx[0:3, 3:6] = np.eye(3) * delta_t
        self.Fx[3:6, 6:9] = -R_mat @ self.vectorToSkewSymmetric(imu_data['acc'] - self.state['a_b']) * delta_t
        self.Fx[3:6, 9:12] = -R_mat * delta_t
        self.Fx[6:9, 6:9] = R.from_rotvec((imu_data['gyr'] - self.state['w_b']) * delta_t).as_matrix()
        self.Fx[6:9, 12:15] = -np.eye(3) * delta_t

    def updateQ(self, delta_t):
        self.Q[0:3, 0:3] *= delta_t**2
        self.Q[3:6, 3:6] *= delta_t**2
        self.Q[6:9, 6:9] *= delta_t
        self.Q[9:12, 9:12] *= delta_t

    def imuPredict(self, imu_data, delta_t):
        self.updateNominalState(imu_data, delta_t)
        self.calcF(imu_data, delta_t)
        self.updateQ(delta_t)
        self.P = self.Fx @ self.P @ self.Fx.T + self.Fi @ self.Q @ self.Fi.T

    def updateH(self):
        X_dx = np.zeros((15, 15))
        X_dx[0:6, 0:6] = np.eye(6)
        X_dx[9:15, 9:15] = np.eye(6)
        R_mat = self.state['R']
        X_dx[6:9, 6:9] = 0.5 * np.array([
            [-R_mat[0, 0], -R_mat[0, 1], -R_mat[0, 2]],
            [ R_mat[0, 0], -R_mat[2, 0],  R_mat[1, 1]],
            [ R_mat[2, 0],  R_mat[0, 0], -R_mat[0, 1]],
            [-R_mat[1, 0],  R_mat[0, 1],  R_mat[0, 0]]
        ])
        self.H = self.Hx @ X_dx

    def updateV(self, uwb_data):
        self.V = uwb_data['cov']

    def uwbUpdate(self, uwb_data, imu_datas, delta_t):
        for imu_data in imu_datas:
            self.imuPredict(imu_data, delta_t)

        xyz = uwb_data['data']
        relative_xyz = xyz - self.ref_position

        self.updateH()
        self.updateV(uwb_data)

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.V)
        dX = K @ (relative_xyz - self.state['p'])
        I_KH = np.eye(15) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.V @ K.T

        self.state['p'] += dX[0:3]
        self.state['v'] += dX[3:6]
        self.state['R'] = self.state['R'] @ R.from_rotvec(dX[6:9]).as_matrix()
        self.state['a_b'] += dX[9:12]
        self.state['w_b'] += dX[12:15]

    def getState(self):
        return self.state

    def vectorToSkewSymmetric(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
    
class ESEKFNode:
    def __init__(self):
        self.ekf = SE3ESEKF()
        ranges = []
        imu = []
        odom = []
        self.ranges_1 = []
        self.ranges_2 = []
        self.ranges_3 = []
        self.ranges_4 = []
        self.ranges_5 = []

        self.timestamp = []
        self.linear_acc_x = []
        self.linear_acc_y = []
        self.linear_acc_z = []
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
            a, b = map(float, line.strip().split("\t"))
            self.gt_x.append(b * -1000)
            self.gt_y.append(a * 1000)
            self.gt_z.append(420)

        self.gt_x, self.gt_y = self.loop_closure(self.gt_x, self.gt_y)
        for line in ranges:
            _, a, b, c, d, e = map(float, line.strip().split("\t"))
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

        imu_datas = []
        for i in range(len(self.timestamp)):
            imu_datas.append({
                'acc': np.array([self.linear_acc_x[i], self.linear_acc_y[i], self.linear_acc_z[i]]),
                'gyr': np.array([self.angular_vel_x[i], self.angular_vel_y[i], self.angular_vel_z[i]])
            })

        self.ekf.imuInit(imu_datas[:100])
        self.ekf.cfgImuVar(0.01, 0.01, 0.01, 0.01)
        self.ekf.cfgRefUwb(0.0, 0.0, 0.0)

        delta_t = 0.07

        for i in range(len(self.timestamp)):
            if self.ranges_1[i] == 0:
                self.ranges_1[i] = self.ranges_1[i - 1]

            if self.ranges_2[i] == 0:
                self.ranges_2[i] = self.ranges_2[i - 1]

            if self.ranges_3[i] == 0:
                self.ranges_3[i] = self.ranges_3[i - 1]

            if self.ranges_4[i] == 0:
                self.ranges_4[i] = self.ranges_4[i - 1]

            if self.ranges_5[i] == 0:
                self.ranges_5[i] = self.ranges_5[i - 1]

            uwb_data = {
                'data': np.array([self.ranges_1[i], self.ranges_2[i], self.ranges_3[i]]),
                'cov': np.eye(3) * 0.1
            }

            self.ekf.uwbUpdate(uwb_data, imu_datas[:i+1], delta_t)
            self.time.append(i)

            self.esti_x.append(self.ekf.getState()["p"][0])
            self.esti_y.append(self.ekf.getState()["p"][1])
            self.esti_z.append(self.ekf.getState()["p"][2])

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

if __name__ == "__main__":
    ESEKFNode()
