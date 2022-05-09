from os import pipe
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path
# from geometry_msgs import PoseStamped 
from nav_msgs.msg import Odometry


import numpy as np

import signal
import sys






def path_cb(data):


	way_point_len = len(data.poses)
	path_length =0

	for i in range(1, way_point_len):

		xval1 = data.poses[i].pose.position.x
		yval1 = data.poses[i].pose.position.y
		zval1 = data.poses[i].pose.position.z 

		X1 = np.asarray([xval1 ,yval1 ,zval1 ])


		xval0 = data.poses[i-1].pose.position.x
		yval0 = data.poses[i-1].pose.position.y
		zval0 = data.poses[i-1].pose.position.z 

		X0 = np.asarray([ xval0 ,yval0 ,zval0 ])

		path_length += np.linalg.norm( X1 - X0 )


	print(path_length)




def listener():

    rospy.init_node('path_commander', anonymous=True)

    print(" INSIDE param estimator ") 
    rospy.Subscriber("/CEMPath" , Path ,path_cb , queue_size= 10 ) ##  CEMPath fastPlanner_path



    rospy.spin()








if __name__ == '__main__':
    listener()