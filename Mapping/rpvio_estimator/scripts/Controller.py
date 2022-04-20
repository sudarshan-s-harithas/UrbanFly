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


path = []
client = airsim.MultirotorClient(ip='10.2.36.227')


def path_cb(path_data):

    global path 
    new_path = []

    print(" INSIDE THE CALLBACK")

    way_point_len = len(path_data.poses)

    print(" ++++++++++++" + str( way_point_len) +" ++++++++ ")

    for i in range( way_point_len):

        xval = path_data.poses[i].pose.position.x
        yval = path_data.poses[i].pose.position.y
        zval = path_data.poses[i].pose.position.z
        # client.moveToPositionAsync(-yval, -xval, -zval, 0.5 , np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate= 0)).join()
 

        # print( xval , yval , zval )
        final = airsim.Vector3r(-yval, -xval, -zval)
        # client.moveToPositionAsync(-yval, -xval, -zval, 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate= np.pi/4)).join()


        way_point = [-yval, -xval, -zval ] #airsim.Vector3r(-yval, -xval, -zval)
        new_path.append(way_point)


    xval = path_data.poses[-1].pose.position.x
    yval = path_data.poses[-1].pose.position.y
    zval = path_data.poses[-1].pose.position.z 

    print( xval , yval , zval )
    # final = airsim.Vector3r(-yval, -xval, -zval)

    # client.moveToPositionAsync(-yval, -xval, -zval, 1.0 , np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate= 0)).join()
    # client.moveOnPathAsync(new_path, 1.25 , np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate= 0)).join()

    path = new_path 
    # control_update_callback()
    # rospy.Timer(rospy.Duration(2), control_update_callback)
    # path.reverse()



def control_update_callback():
    goal = path[int(len(path)/2)]
    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position

    # goal_distance = 1
    # if len(path) < 5:
    #     goal_distance = 0.5

    # for way_point in path:
    #     if (way_point - current_position).get_length() <= goal_distance:
    #         goal = way_point

    # displacement = goal - current_position
    # direction = displacement / displacement.get_length()

    numPts = len(path)

    print( numPts)


    # for i in range( 1,numPts ):

    #     vx = path[i][0] - path[i -1][0]
    #     vy = path[i][1] - path[i -1][1]
    #     vz = path[i][2] - path[i-1][2]

    t = abs( path[-1][0] - path[0][0] + path[-1][1] - path[0][1] + path[-1][2] - path[0][2] )/3


    vx = path[-1][0] - path[0][0]
    vy = path[-1][1] - path[0][1]
    vz = path[-1][2] - path[0][2]

    vx /= np.linalg.norm( [vx , vy, vz]  )
    vy /= np.linalg.norm( [vx , vy, vz]  )
    vz /= np.linalg.norm( [vx , vy, vz]  )

    client.moveByVelocityAsync(vx*0.3 , vy*0.3 , 0, 10 , airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, np.pi/4))




def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'path_commander' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('path_commander', anonymous=True)

    # rospy.Subscriber("/feasible_path", PointCloud, path_update_callback)
    print(" INSIDE listener ")
    rospy.Subscriber("CEMPath" , Path ,path_cb , queue_size= 10 )


    # rospy.Timer(rospy.Duration(2), control_update_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()