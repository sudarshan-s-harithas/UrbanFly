#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from sensor_msgs.msg import PointCloud

import airsim

import signal
import sys
import time

from threading import Lock
import numpy as np

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    client.reset()

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

# Global variables
odom_lock = Lock()
latest_odom_msg = None
vins_airsim_offset = airsim.Vector3r(0, 0, 0)
inited = False

client = airsim.MultirotorClient(ip='10.2.36.227')
client.confirmConnection()
client.enableApiControl(True)
client.takeoffAsync(timeout_sec=10)

path = []
path.append(airsim.Vector3r(0.0, 0.0, 0))
path.append(airsim.Vector3r(0.0, 0.0, -1))
path.append(airsim.Vector3r(0.0, 0.0, -3))
path.append(airsim.Vector3r(0.0, 0.0, -1))
path.append(airsim.Vector3r(0.0, 0.0, -2.5))
client.moveByRollPitchYawZAsync(0.0, 0.0, np.pi/2, -0.35, 2).join()
client.moveOnPathAsync(path, 0.4, 10, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
client.moveByRollPitchYawrateZAsync(0.0, 0.0, -np.pi/64, -3, 15).join()
#client.moveByRollPitchYawrateZAsync(0.0, 0.0, np.pi/64, -3, 4).join()

prev_goal = None

def get_current_position():
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position
    
    print("current airsim position")
    print(current_position)
    return current_position

def latest_odom_position():
    odom_lock.acquire()
    current_odom = latest_odom_msg
    odom_lock.release()
    position = current_odom.pose.pose.position
    position_av = airsim.Vector3r(-position.y, -position.x, -position.z)
    print("current odom position")
    print(position_av)
    return position_av

def move_by_delta(delta):
    p = get_current_position() + delta
    client.moveToPositionAsync(p.x_val, p.y_val, p.z_val, 0.5, 5, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0))
    rospy.sleep(10)

def move_on_pattern():
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position
    print("Current position is ", current_position)

    x1 = airsim.Vector3r(2.0, 0.0, 0.0)
    y1 = airsim.Vector3r(0.0, 2.0, 0.0)
    z1 = airsim.Vector3r(0.0, 0.0, 2.0)

    global vins_airsim_offset
    rospy.sleep(10)
    vins_airsim_offset = get_current_position() - latest_odom_position()
    move_by_delta(x1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    move_by_delta(x1 * -1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    move_by_delta(y1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    move_by_delta(y1 * -1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    move_by_delta(z1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    move_by_delta(z1 * -1)
    vins_airsim_offset += get_current_position() - latest_odom_position()
    
    vins_airsim_offset /= 7
    
    print("VINS - Airsim offset is ")
    print(vins_airsim_offset)
    
    # client.moveOnPathAsync(pattern, 0.5, 20, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()

def vins_odom_callback(odom_msg):
    global latest_odom_msg
    odom_lock.acquire()
    latest_odom_msg = odom_msg
    odom_lock.release()
    #print("Received odom ", latest_odom_position())

#path = []
def path_update_callback(nav_path):
    print("Received new path")
    global path
    new_path = []
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position
    
    for nav_pose in nav_path.poses:
        way_pt = nav_pose.pose.position
        X = np.array([[-way_pt.y], [-way_pt.x]])
        R = np.array([ [1, 0], [0, 1]])
        x = (R @ X).flatten()
        way_point = airsim.Vector3r(x[0], x[1], -way_pt.z) + vins_airsim_offset
        # if (way_point - current_position).get_length() > 3:
        new_path.append(way_point)
    # retain the points only from 5m away from current position
    rev_path = new_path[::-1]
    rev_path2 = []
    for rpt in rev_path:
        if (rpt - current_position).get_length() < 4:
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
        client.moveToPositionAsync(goal.x_val, goal.y_val, goal.z_val, 0.5, 10, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0))

def listener():


    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'path_commander' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('path_commander', anonymous=True)

    rospy.Subscriber("/feasible_path2", Path, path_update_callback)
    rospy.Subscriber("/vins_estimator/odometry", Odometry, vins_odom_callback)

    rospy.Timer(rospy.Duration(secs=1), control_update_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.sleep(5) # Sleep for 5 seconds, to allow vins to stabilize
    move_on_pattern()
    rospy.sleep(5)
    rospy.spin()

if __name__ == '__main__':
    listener()
