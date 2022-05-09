from os import pipe
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud
from nav_msgs.msg import Path


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
client.reset()
client.confirmConnection()
client.enableApiControl(True)
client.takeoffAsync()
client.hoverAsync()

client.moveToZAsync(-2, 0.5)
client.moveToZAsync(0, 0.5)
client.moveToZAsync(-2.5, 0.5)

print(" Vertical Movement over ")

'''
path = []
path.append(airsim.Vector3r(0.0, 0.0, -5))
# path.append(airsim.Vector3r(0.0, 0.0, -1))
# path.append(airsim.Vector3r(0.0, 0.0, -5))

client.moveOnPathAsync(path, 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join() # 0.5 yaw mode

path = []

path.append(airsim.Vector3r( 0, -5 , -5))
# path.append(airsim.Vector3r( 0, -1 , -5))
# path.append(airsim.Vector3r(0 , -5, -5))
path.append(airsim.Vector3r(0.0, -5, -1))
path.append(airsim.Vector3r(0.0, -5, -3))


client.moveOnPathAsync(path, 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join() # 0.5 yaw mode

# client.enableApiControl(False)
'''
pt1 = [ 0.0, 0.0, -3 ]
pt2 = [0.0, 0.0, -4  ]
pt3= [ 0, 0.0 , -10  ]
# pt3 = [0,0,0 ]
pt4 = [  0, -5 , -5 ]
pt5= [  0, -5 , -5]

client.moveToPositionAsync( pt1[0] , pt1[1] , pt1[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0)).join()
# client.moveByRollPitchYawrateZAsync(0.0, 0.0, -np.pi/64, -2.75, 12).join()
client.moveToPositionAsync( pt2[0] , pt2[1] , pt2[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt1[0] , pt1[1] , pt1[2] , 1.4, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt2[0] , pt2[1] , pt2[2] , 0.4, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt1[0] , pt1[1] , pt1[2] , 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt2[0] , pt2[1] , pt2[2] , 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt3[0] , pt3[1] , pt3[2] , 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.rotateByYawRateAsync( -0.5 , 3 )
# client.rotateByYawRateAsync( 0.5 , 3)
# client.moveToPositionAsync( pt4[0] , pt4[1] , pt4[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()
# client.moveToPositionAsync( pt5[0] , pt5[1] , pt5[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()

print(" Initilizer Code Complete ")