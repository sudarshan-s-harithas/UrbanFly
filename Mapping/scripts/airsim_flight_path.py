import numpy as np
import airsim

import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    client.reset()

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

client = airsim.MultirotorClient(ip='10.2.36.227')
client.confirmConnection()
client.reset()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.hoverAsync().join()
# client.moveToZAsync(3, 3).join()

# yaw the drone to face the camera
# client.moveByAngleRatesZAsync(0.0, 0.0, np.pi/2, -1, 1).join()

path = []

path.append(airsim.Vector3r(0.0, -22.0, -2.5))
path.append(airsim.Vector3r(22.0, -22.0, -2.5))
path.append(airsim.Vector3r(22.0, 0.0, -2.5))
path.append(airsim.Vector3r(0.0, 0.0, -2.5))

# path.append(airsim.Vector3r(225.0, -225.0, 3))
# path.append(airsim.Vector3r(-90.0, 95.0, 3))
# path.append(airsim.Vector3r(22.0, 0.0, -25))
# path.append(airsim.Vector3r(0.0, 0.0, -20))

# path.append(airsim.Vector3r())
# path.append(airsim.Vector3r())
# path.append(airsim.Vector3r())
# path.append(airsim.Vector3r())
# path.append(airsim.Vector3r())
# path.append(airsim.Vector3r())

client.moveToZAsync(-4, 0.5).join()
# client.moveToZAsync(-7, 0.5).join()
# client.moveToZAsync(-2, 0.5).join()
client.moveToZAsync(0, 0.5).join()
# client.moveToZAsync(-1, 0.5).join()
client.moveToZAsync(-2.5, 0.5).join()
# client.moveToPositionAsync(0.0, -20.0, -2, 1).join()

print("flying on smooth path..")
client.moveOnPathAsync(path, 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, np.pi/4)).join()

# End motion and recording
print("Done ...")
# client.moveToZAsync(5, 2).join()
# client.moveToPositionAsync(0, 0, 0, 1).join()
client.landAsync().join()
client.armDisarm(False)
client.reset()
client.enableApiControl(False)
