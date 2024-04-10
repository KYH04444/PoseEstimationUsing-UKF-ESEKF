import rospy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from uwb_imu.msg import UwbMsg
class Plotter:
    def __init__(self):
        self.esekf_x_uwb = []
        self.esekf_y_uwb = []
        self.esekf_z_uwb = []

        self.ukf_x_uwb = []
        self.ukf_y_uwb = []
        self.ukf_z_uwb = []

        self.x_local = []
        self.y_local = []
        self.z_local = []
        self.esekf_uwb_cnt = 0
        self.ukf_uwb_cnt = 0
        self.imu_cnt = 0
        rospy.init_node("plotter")
        rospy.Subscriber("/result", PoseStamped, self.esekf_uwb_callback)
        rospy.Subscriber("/result_ukf", PoseStamped, self.ukf_uwb_callback)
        rospy.Subscriber("/uwb_filtered", UwbMsg, self.local_callback)

    def esekf_uwb_callback(self, msg):
        self.esekf_uwb_cnt+=1
        if self.esekf_uwb_cnt >= 3:
            self.esekf_x_uwb.append(msg.pose.position.x)
            self.esekf_y_uwb.append(msg.pose.position.y)
            self.esekf_z_uwb.append(msg.pose.position.z)
            # print("ing")
            self.esekf_uwb_cnt = 0

    def ukf_uwb_callback(self, msg):
        self.ukf_uwb_cnt+=1
        if self.ukf_uwb_cnt >= 3:
            self.ukf_x_uwb.append(msg.pose.position.x)
            self.ukf_y_uwb.append(msg.pose.position.y)
            self.ukf_z_uwb.append(msg.pose.position.z)
            # print("ing")
            self.ukf_uwb_cnt = 0

    def local_callback(self, msg):
        self.imu_cnt+=1
        if self.imu_cnt >= 3:
            self.x_local.append(msg.pos_x)
            self.y_local.append(msg.pos_y)
            self.z_local.append(msg.pos_z)
            self.imu_cnt = 0
    
    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while not rospy.is_shutdown():
            ax.clear()

            ax.scatter(self.esekf_x_uwb, self.esekf_y_uwb, self.esekf_z_uwb, c='r', label='ESEKF', s=1)
            ax.scatter(self.ukf_x_uwb, self.ukf_y_uwb, self.ukf_z_uwb, c='g', label='UKF', s=1)
            ax.scatter(self.x_local, self.y_local, self.z_local, c='b', label='GT',s=5)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            # ax.set_zlabel('Z')
            ax.set_title('Real-Time 3D Plot')
            ax.legend()

            plt.pause(0.1)
            rospy.sleep(0.1)

if __name__ == "__main__":
    plotter = Plotter()
    plotter.plot_data()
