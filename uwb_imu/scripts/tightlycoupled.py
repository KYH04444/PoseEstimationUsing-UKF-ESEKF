import numpy as np
import matplotlib.pyplot as plt
import math as m
v_i = 0.1
w_i = 0.063
v_j = 0.139
w_j = 0.09


class TightlyCoupledEKF:
    def __init__(self):
        self.prev_m_vecX_2 = 0
        self.m_vecX = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        self.m_matP = np.zeros((5, 5), dtype=np.float32)
        self.B = np.zeros((5,2), dtype=np.float32)
        self.m_matQ = 0.001 * np.eye(5, dtype=np.float32)
        self.m_jacobian_matF = np.zeros((5, 5), dtype=np.float32)
        self.m_vecZ = np.zeros(5, dtype=np.float32)
        self.m_vech = np.zeros(5, dtype=np.float32)
        self.m_matR = 0.1 * np.eye(5, dtype=np.float32)
        self.m_jacobian_matH = np.zeros((5, 5), dtype=np.float32)
        self.rotation_m = np.ones((2,2), dtype=np.float32)
        self.uwb_position = np.array([[4050, -3900, 1200, -900, 4200],
                                     [2550, 2400, 3000, -3600, -3000],
                                     [2270, 1290, 430, 670, 1120]], dtype=np.float32)
    def getVecX(self):
        return self.m_vecX
    
    def setIMU(self, msg, delta_t):
        self.linear_acc_x = msg[0]  
        self.linear_acc_y = msg[1]  
        self.angular_vel_x = msg[2]  
        self.angular_vel_y = msg[3]      
        self.angular_vel_z = msg[4]
        self.rotation_m   =[[m.cos(self.angular_vel_z*delta_t*180/m.pi),-m.sin(self.angular_vel_z*delta_t*180/m.pi)],
                            [m.sin(self.angular_vel_z*delta_t*180/m.pi), m.cos(self.angular_vel_z*delta_t*180/m.pi)]]
        
    def getMatP(self):
        return self.m_matP

    def getvecZ(self):
        return self.m_vecZ

    def motionModelJacobian(self, delta_t):
        self.m_jacobian_matF = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]], dtype=np.float32)
        self.m_jacobian_matF = np.eye(5) + delta_t * self.m_jacobian_matF

        self.B = np.array([
            [0.5*delta_t**2, 0],
            [0, 0.5*delta_t**2],
            [delta_t, 0],
            [0, delta_t],
            [0, 0]], dtype=np.float32)
        
    def motionModel(self, delta_t):
        self.m_vecX = np.dot(self.m_jacobian_matF,self.m_vecX) + \
                      np.dot(np.dot(self.B, self.rotation_m), np.array([self.linear_acc_x, self.linear_acc_y])) + \
                      np.array([0, 0, 0, 0, self.angular_vel_z*180/m.pi*delta_t])

    def prediction(self, delta_t):
        self.motionModelJacobian(delta_t)
        self.motionModel(delta_t)
        self.m_matP = np.dot(np.dot(self.m_jacobian_matF, self.m_matP), self.m_jacobian_matF.T) + self.m_matQ

    def measurementModel(self): 
        self.m_vech[0] =np.sqrt((self.m_vecX[0]- self.uwb_position[0][0])**2 +(self.m_vecX[1]- self.uwb_position[1][0])**2)
        self.m_vech[1] =np.sqrt((self.m_vecX[0]- self.uwb_position[0][1])**2 +(self.m_vecX[1]- self.uwb_position[1][1])**2)
        self.m_vech[2] =np.sqrt((self.m_vecX[0]- self.uwb_position[0][2])**2 +(self.m_vecX[1]- self.uwb_position[1][2])**2)
        self.m_vech[3] =np.sqrt((self.m_vecX[0]- self.uwb_position[0][3])**2 +(self.m_vecX[1]- self.uwb_position[1][3])**2)
        self.m_vech[4] =np.sqrt((self.m_vecX[0]- self.uwb_position[0][4])**2 +(self.m_vecX[1]- self.uwb_position[1][4])**2)

    def measurementModelJacobian(self):
        self.m_jacobian_matH = np.array([[(self.m_vecX[0]- self.uwb_position[0][0])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][0])**2 +(self.m_vecX[1]- self.uwb_position[1][0])**2), (self.m_vecX[1]- self.uwb_position[1][0])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][0])**2 +(self.m_vecX[1]- self.uwb_position[1][0])**2),0,0,0],
                                         [(self.m_vecX[0]- self.uwb_position[0][1])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][1])**2 +(self.m_vecX[1]- self.uwb_position[1][1])**2), (self.m_vecX[1]- self.uwb_position[1][1])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][1])**2 +(self.m_vecX[1]- self.uwb_position[1][1])**2),0,0,0],
                                         [(self.m_vecX[0]- self.uwb_position[0][2])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][2])**2 +(self.m_vecX[1]- self.uwb_position[1][2])**2), (self.m_vecX[1]- self.uwb_position[1][2])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][2])**2 +(self.m_vecX[1]- self.uwb_position[1][2])**2),0,0,0],
                                         [(self.m_vecX[0]- self.uwb_position[0][3])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][3])**2 +(self.m_vecX[1]- self.uwb_position[1][3])**2), (self.m_vecX[1]- self.uwb_position[1][3])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][3])**2 +(self.m_vecX[1]- self.uwb_position[1][3])**2),0,0,0],
                                         [(self.m_vecX[0]- self.uwb_position[0][4])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][4])**2 +(self.m_vecX[1]- self.uwb_position[1][4])**2), (self.m_vecX[1]- self.uwb_position[1][4])/np.sqrt((self.m_vecX[0]- self.uwb_position[0][4])**2 +(self.m_vecX[1]- self.uwb_position[1][4])**2),0,0,0]]
                                        , dtype=np.float32)

    def correction(self):
        self.measurementModel()
        self.measurementModelJacobian()

        residual = self.m_vecZ - self.m_vech

        residual_cov = np.dot(np.dot(self.m_jacobian_matH, self.m_matP), self.m_jacobian_matH.T) + self.m_matR


        Kk = np.dot(np.dot(self.m_matP, self.m_jacobian_matH.T), np.linalg.inv(residual_cov))

        self.m_vecX += np.dot(Kk, residual)
        self.m_matP = np.dot((np.eye(5) - np.dot(Kk, self.m_jacobian_matH)), self.m_matP)



class EKFNode:
    def __init__(self):
        self.ekf_tightly_coupled = TightlyCoupledEKF()
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
        self.angular_vel_x = []
        self.angular_vel_y = []
        self.angular_vel_z = []

        self.rms_x = []
        self.rms_y = []
        self.rms_t = []

        self.gt_x = []
        self.gt_y = []
        self.gt_theta = []
        self.gt_v = []
        self.gt_w = []

        self.ceres_xji = []
        self.ceres_yji = []
        
        self.esti_x = []
        self.esti_y = []
        self.esti_theta = []
        self.esti_v_x = []
        self.esti_v_y = []
        self.esti_w = []
        self.time = []
        with open('uwb_ranges.txt', 'r') as file:
            ranges = file.readlines()
        with open('odom.txt', 'r') as file:
            odom = file.readlines()

        for line in odom:
            a, b =map(float, line.strip().split("\t"))
            self.gt_x.append(a*1000)
            self.gt_y.append(b*-1000)

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
            a, b, c, d, e, f =map(float, line.strip().split("\t"))
            self.timestamp.append(a)
            self.linear_acc_x.append(b)
            self.linear_acc_y.append(c)
            self.angular_vel_x.append(d)
            self.angular_vel_y.append(e)
            self.angular_vel_z.append(f)

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

            self.ekf_tightly_coupled.getvecZ()[0] = self.ranges_1[i] 
            self.ekf_tightly_coupled.getvecZ()[1] = self.ranges_2[i] 
            self.ekf_tightly_coupled.getvecZ()[2] = self.ranges_3[i]
            self.ekf_tightly_coupled.getvecZ()[3] = self.ranges_4[i]
            self.ekf_tightly_coupled.getvecZ()[4] = self.ranges_5[i]
            
            self.ekf_tightly_coupled.setIMU(np.array([self.linear_acc_x[i], self.linear_acc_y[i],self.angular_vel_x[i],self.angular_vel_y[i],self.angular_vel_z[i]]), delta_t) 

            self.ekf_tightly_coupled.prediction(delta_t)
            self.ekf_tightly_coupled.correction()
            self.time.append(i)

            self.esti_x.append(self.ekf_tightly_coupled.getVecX()[0])
            self.esti_y.append(self.ekf_tightly_coupled.getVecX()[1])
            self.esti_v_y.append(self.ekf_tightly_coupled.getVecX()[2])
            self.esti_v_y.append(self.ekf_tightly_coupled.getVecX()[3])
            self.esti_theta.append(self.ekf_tightly_coupled.getVecX()[4])
        
        # print(np.sqrt((np.mean(self.rms_x))))
        # print(np.sqrt((np.mean(self.rms_y))))
        # print(np.sqrt((np.mean(self.rms_t))))
        plt.plot(self.esti_x, self.esti_y, linestyle = '-', color ='red')

        plt.plot(self.gt_y, self.gt_x, linestyle = '-', color ='black')
        # fig, axs  =plt.subplots(2,1, figsize=(10,12))

        # axs[0].set_ylabel('X ji [m]')
        # # axs[0].plot(self.gt_xji,linestyle='-', color='blue',label = 'true x')
        # axs[0].plot(self.esti_x, linestyle='-',color='red',label = 'ekf x')
        # axs[0].grid(True)

        # axs[1].set_ylabel('Y ji [m]')
        # # axs[1].plot( self.gt_yji,linestyle='-', color='blue',label = 'true y')
        # axs[1].plot( self.esti_y,linestyle='-', color='red',label = 'ekf y')
        # axs[1].grid(True)
        # for ax in axs:
        #     ax.legend()
        plt.show()
        with open("UWB_IMU.txt", 'w') as file:
            for x,y,z,a,b  in zip(self.esti_x, self.esti_y, self.esti_v_y, self.esti_v_y, self.esti_theta):
                file.write(f"{x}\t {y}\t{z}\t{a}\t{b}\n")
                


if __name__ =="__main__":
    EKFNode()