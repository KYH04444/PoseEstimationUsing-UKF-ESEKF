import rospy
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from uwb_imu.msg import UwbMsg
class Plotter:
    def __init__(self):
        self.x_uwb = []
        self.y_uwb = []
        self.z_uwb = []
        self.x_local = []
        self.y_local = []
        self.z_local = []
        self.uwb_cnt = 0
        self.imu_cnt = 0
        rospy.init_node("plotter")
        rospy.Subscriber("/result", PoseStamped, self.uwb_callback)
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_callback)

    def uwb_callback(self, msg):
        self.uwb_cnt+=1
        if self.uwb_cnt >= 3:
            self.x_uwb.append(msg.pose.position.x)
            self.y_uwb.append(msg.pose.position.y)
            self.z_uwb.append(msg.pose.position.z)
            # print("ing")
            self.uwb_cnt = 0

    def local_callback(self, msg):
        self.imu_cnt+=1
        if self.imu_cnt >= 3:
            self.x_local.append(msg.pose.position.x)
            self.y_local.append(msg.pose.position.y)
            self.z_local.append(msg.pose.position.z)
            self.imu_cnt = 0
    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        while not rospy.is_shutdown():
            ax.clear()

            ax.scatter(self.x_uwb, self.y_uwb, self.z_uwb, c='r', label='ESEKF', s=1)
            ax.scatter(self.x_local, self.y_local, self.z_local, c='b', label='GT',s=5)

            # ax.scatter(self.x_uwb, self.y_uwb, c='r', label='UWB Filtered', s=1)
            # ax.scatter(self.x_local, self.y_local,  c='b', label='Local Position',s=5)

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
