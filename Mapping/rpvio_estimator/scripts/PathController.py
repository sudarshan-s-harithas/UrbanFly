from os import pipe
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path
# from geometry_msgs import PoseStamped 
from nav_msgs.msg import Odometry

import airsim
import cvxpy
import numpy as np

import signal
import sys

path = []
client = airsim.MultirotorClient(ip='10.2.36.227')


def PathExecute(event):

	goal = path.pop(3) ## Take one in 3 points 

	client.moveToPositionAsync(goal[0],goal[1], -(goal[2]+1.2), 0.1, 5, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0))




def path_cb(path_data):

	way_point_len = len(path_data.poses)

	for i in range(way_point_len):

		yval = -path_data.poses[-1].pose.position.x
		xval = -path_data.poses[-1].pose.position.y
		zval = path_data.poses[-1].pose.position.z 
		R = np.array( [ [ 0.707 , -0.707  ] , [ 0.707 , 0.707] ] )

		X = np.array( [ [ xval] , [yval] ])
		xval , yval =  (R@X).flatten() 
		path.append( [ xval , yval , zval ])




def listener():

    rospy.init_node('path_commander', anonymous=True)

    print(" INSIDE listener ") 
    rospy.Subscriber("/CEMPath" , Path ,path_cb , queue_size= 10 ) ##  CEMPath fastPlanner_path
    rospy.Subscriber("/vins_estimator/odometry" , Odometry ,VinsState_cb , queue_size= 10 )


    rospy.Timer(rospy.Duration(1), PathExecute)

    rospy.spin()


if __name__ == '__main__':
    listener()