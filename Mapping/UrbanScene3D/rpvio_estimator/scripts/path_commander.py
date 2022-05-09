#!/usr/bin/env python
from doctest import FAIL_FAST
from mimetypes import init
from os import pipe
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud

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
#client.takeoffAsync(timeout_sec=10)
# client.hoverAsync()

#client.moveToZAsync(-2, 0.5)
#client.moveToZAsync(0, 0.5)
#client.moveToZAsync(-2.5, 0.5)

path = []
path.append(airsim.Vector3r(0.0, 0.0, 0))
path.append(airsim.Vector3r(0.0, 0.0, -1))
path.append(airsim.Vector3r(0.0, 0.0, -3))
path.append(airsim.Vector3r(0.0, 0.0, -1))
path.append(airsim.Vector3r(0.0, 0.0, -2.5))
client.moveByRollPitchYawZAsync(0.0, 0.0, np.pi/2, -0.35, 2).join()
client.moveOnPathAsync(path, 0.4, 10, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
#client.moveByRollPitchYawrateZAsync(0.0, 0.0, -np.pi/64, -2.75, 12).join()
#client.moveByRollPitchYawrateZAsync(0.0, 0.0, np.pi/64, -3, 4).join()

prev_goal = None

#path = []
def path_update_callback(way_points):
    print("Received new path")
    global path
    new_path = []
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position
    
    for way_pt in way_points.points:
        X = np.array([[-way_pt.y], [-way_pt.x]])
        R = np.array([ [1, 0], [0, 1]])
        x = (R @ X).flatten()
        way_point = airsim.Vector3r(x[0], x[1], -5)
        # if (way_point - current_position).get_length() > 3:
        new_path.append(way_point)
    # retain the points only from 5m away from current position
    rev_path = new_path[::-1]
    rev_path2 = []
    for rpt in rev_path:
        if (rpt - current_position).get_length() < 2:
            break
        rev_path2.append(rpt)

    #if len(new_path) >= 3:
    #    path = new_path
    path = rev_path2[::-1]
    global inited
    inited = True

def control_update_callback(event):
    global path
    if not inited:
        print("not initialized")
    elif len(path) < 1:
        print("no waypoints in the path")
    else:
        goal = path.pop(0)
        client.moveToPositionAsync(goal.x_val, goal.y_val, -5, 0.75, 5, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.65))

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'path_commander' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('path_commander', anonymous=True)

    rospy.Subscriber("/feasible_path", PointCloud, path_update_callback)

    rospy.Timer(rospy.Duration(secs=1), control_update_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
