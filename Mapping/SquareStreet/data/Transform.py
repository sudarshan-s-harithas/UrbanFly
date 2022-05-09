import numpy as np
import time
from nav_msgs.msg import Path
import rospy
from geometry_msgs.msg import PoseStamped
import time 
import std_msgs.msg
import time 

path = np.loadtxt('SquareStreetDemoPath.txt')

R = np.array([
    [   0.707,  -0.707],
	[   0.707,  0.707 ]
])

# nav_path = Path()

cnt =0 

path_pub = rospy.Publisher("/CEMPath_transform2" , Path, queue_size=10 ) ##  CEMPath fastPlanner_path


def RUNPUB():
	cnt =0 

	CEM_Path = Path()

	h = std_msgs.msg.Header()
	h.stamp = rospy.Time.now() # Note you need to call rospy.init_node() before this will work
	h.frame_id = "world"
	global path_pub
	CEM_Path.header =h
	for pose in path[10:]:
		pose_path = PoseStamped()
		pose_path.header = h

		pt_2d = np.array([[pose[0]],[pose[1]]])
		x_val, y_val = (R.T).dot(pt_2d).flatten()
		z_val = pose[-1]
		pose_path.pose.position.x = x_val + 9.0
		pose_path.pose.position.y = y_val
		pose_path.pose.position.z = z_val

		CEM_Path.poses.append(pose_path)
		
		if( cnt %25 == 0):
			key_ = input(" Press for next  ")
			print(cnt )
			path_pub.publish( CEM_Path )
			# time.sleep(2)
		cnt +=1 

def listener():

	rospy.init_node('CEMPath_transform', anonymous=True)

	RUNPUB()

	rospy.spin()




if __name__ == '__main__':
    listener()
