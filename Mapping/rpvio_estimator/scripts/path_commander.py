#!/usr/bin/env python
from os import pipe
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path
# from geometry_msgs import PoseStamped 

import airsim
import cvxpy
import numpy as np

import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    client.reset()

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

inited = False
client = airsim.MultirotorClient(ip='10.2.36.227')
client.confirmConnection()
client.enableApiControl(True)
client.takeoffAsync()
client.hoverAsync()

client.moveToZAsync(-2, 0.5)
client.moveToZAsync(0, 0.5)
client.moveToZAsync(-2.5, 0.5)

path = []
path.append(airsim.Vector3r(0.0, 0.0, -5))
path.append(airsim.Vector3r(0.0, 0.0, -1))
path.append(airsim.Vector3r(0.0, 0.0, -5))
client.moveOnPathAsync(path, 2.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.5)).join()

prev_goal = None

def cruisingToPrevGoal(goal, client):
    if goal is None:
        return False
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position

    return (goal - current_position).get_length() > 1

def callback(pcd):
    if len(pcd.points) < 5:
        goal_pt = pcd.points[-1]
    else:
        mid = int(len(pcd.points)/2)
        goal_pt = pcd.points[mid]
    goal = airsim.Vector3r(-goal_pt.y, -goal_pt.x, -5)
    
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position

    displacement = goal - current_position
    direction = displacement / displacement.get_length()
    client.moveByVelocityAsync(direction.x_val*0.25, direction.y_val*0.25, 0.0, 5, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.5))
    #rospy.sleep(duration=5)

#path = []
def path_update_callback(way_points):
    print("Received new path")
    global path
    new_path = []
    #current_state = client.getMultirotorState()
    #current_position = current_state.kinematics_estimated.position
    
    for way_pt in way_points.points:
        way_point = airsim.Vector3r(-way_pt.y, -way_pt.x, -5)
        # if (way_point - current_position).get_length() > 3:
        new_path.append(way_point)
    #if len(new_path) >= 3:
    #    path = new_path
    path = new_path 
    path.reverse()

def control_update_callback(event):
    goal = path[int(len(path)/2)]
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position

    goal_distance = 3
    if len(path) < 5:
        goal_distance = 0.5

    for way_point in path:
        if (way_point - current_position).get_length() <= goal_distance:
            goal = way_point

    displacement = goal - current_position
    direction = displacement / displacement.get_length()

    client.moveByVelocityAsync(direction.x_val*0.3, direction.y_val*0.3, displacement.z_val*0.3, 5, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.5))




def path_cb(path_data):

    global path 
    new_path = []

    print(" INSIDE THE CALLBACK")

    way_point_len = len(path_data.poses)

    for i in range( way_point_len):

        xval = path_data.poses[i].pose.position.x
        yval = path_data.poses[i].pose.position.y
        zval = path_data.poses[i].pose.position.z 

        print( xval , yval , zval )

        way_point = airsim.Vector3r(-yval, -xval, -zval)
        new_path.append(way_point)

    # client.moveOnPathAsync(new_path, 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate=0.5)).join()

    path = new_path 
    # path.reverse()




def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'path_commander' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('path_commander', anonymous=True)

    # rospy.Subscriber("/feasible_path", PointCloud, path_update_callback)
    print(" INSIDE listener ")
    rospy.Subscriber("CEMPath" , Path ,path_cb )


    rospy.Timer(rospy.Duration(2), control_update_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
