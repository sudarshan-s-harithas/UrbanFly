"""
This is the python version of planner_node.cpp
"""
#!/usr/bin/env python
import rospy
from rospy.rostime import Time
from std_msgs.msg import String
import message_filters
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

from scipy.spatial.transform.rotation import Rotation
from planner import Planner
import numpy as np

local_goal_pub = None
local_stomp_pub = None
feasible_path_pub = None
free_cloud_pub = None
colliding_cloud_pub = None

is_inited = False
prev_msg = None
replan_distance = 5

def should_replan(odom_msg1, odom_msg2):
    # Compute distance
    trans1 = odom_msg1.pose.pose.position
    pos1 = np.array([trans1.x, trans1.y, 0]).reshape((3, 1))
    trans2 = odom_msg2.pose.pose.position
    pos2 = np.array([trans2.x, trans2.y, 0]).reshape((3, 1))
    distance = np.linalg.norm(pos1 - pos2)
    return distance >= replan_distance

def vertices_and_odom_callback(vertices_msg, odometry_msg):
    global local_goal_pub
    global local_stomp_pub
    global feasible_path_pub
    global free_cloud_pub
    global colliding_cloud_pub
    global is_inited
    global prev_msg
    
    if not is_inited or should_replan(odometry_msg, prev_msg):
        planner = Planner(vertices_msg, odometry_msg, local_goal_pub, local_stomp_pub, feasible_path_pub, free_cloud_pub, colliding_cloud_pub)
        planner.compute_paths()
        planner.publish_paths()
        prev_msg = odometry_msg
        is_inited = True
        print("New plan !")
    else:
        print("Move atleast ", replan_distance, "m to replan")

def register_pub_sub():
    rospy.init_node('pyplanner', anonymous=True)

    vertices_sub = message_filters.Subscriber("/rpvio_mapper/vertices", PointCloud, queue_size=10)
    odometry_sub = message_filters.Subscriber("/vins_estimator/odometry", Odometry, queue_size=10)
    ts = message_filters.TimeSynchronizer([vertices_sub, odometry_sub], 20)
    ts.registerCallback(vertices_and_odom_callback)

    global local_goal_pub
    global local_stomp_pub
    global feasible_path_pub
    global free_cloud_pub
    global colliding_cloud_pub

    local_goal_pub = rospy.Publisher("local_goal", PointCloud, queue_size=10)
    local_stomp_pub = rospy.Publisher("gaussian_paths", MarkerArray, queue_size=1)
    feasible_path_pub = rospy.Publisher("feasible_path", PointCloud, queue_size=5)
    free_cloud_pub = rospy.Publisher("free_cloud", PointCloud, queue_size=20)
    colliding_cloud_pub = rospy.Publisher("colliding_cloud", PointCloud, queue_size=20)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    register_pub_sub()
