import numpy as np
import airsim

import signal
import sys
import time

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    client.reset()

signal.signal(signal.SIGINT, signal_handler)
print('Press Ctrl+C')

client = airsim.MultirotorClient(ip='10.2.36.169')
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

# path.append(airsim.Vector3r(0.0, -22.0, -5))
# path.append(airsim.Vector3r(22.0, -22.0, -5))
# path.append(airsim.Vector3r(22.0, 0.0, -5))
# path.append(airsim.Vector3r(0.0, 0.0, -5))

# path.append(airsim.Vector3r(75.0, -90, -5))
# path.append(airsim.Vector3r(100.0, 50.0, -10))
# path.append(airsim.Vector3r(0.0, 50.0, -5))
# path.append(airsim.Vector3r(0.0, 0.0, -5))

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

# client.moveToZAsync(-5, 1, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=False, yaw_or_rate=-45)).join()
# client.moveToZAsync(-7, 0.5).join()
# client.moveToZAsync(-2, 0.5).join()
# client.moveToZAsync(0, 1).join()
# client.moveToZAsync(-1, 0.5).join()
# client.moveToZAsync(-6, 0.5).join()
# client.moveToZAsync(0, 0.5).join()
# client.moveToZAsync(-5, 1).join()
# client.moveToZAsync(0, 0.5).join()
# client.moveToPositionAsync(0.0, -20.0, -2, 1).join()

print("Performing up-down motion")
# client.moveToZAsync(-4, 0.5).join()
# client.moveToZAsync(-7, 0.5).join()
client.moveToZAsync(-2, 0.5).join()
client.moveToZAsync(-2.5, 0.5).join()

print("Starting to execute the trajectory")
mid_position = airsim.Vector3r(0.0, -22.0, -2.5)
end_position = airsim.Vector3r(22.0, -22.0, -2.5)

# path.append(airsim.Vector3r(25.0, 0, -5))
# client.moveOnPathAsync(path, 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate=-0.5))
# path = []
# path.append(airsim.Vector3r(50.0, -20, -7))
# client.moveOnPathAsync(path, 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate=-0.5))
# path = []
# path.append(airsim.Vector3r(77.0, -90, -5))
# client.moveOnPathAsync(path, 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate=0.0)).join()

# Arm the drone, take off and dance

current_state = client.getMultirotorState()
current_position = current_state.kinematics_estimated.position
# end_position = airsim.Vector3r(55.0, 0, -5)
# print('end_position: ', end_position.x_val, end_position.y_val, end_position.z_val)

# client.simPause(True)
while (end_position - current_position).get_length() > 1:
    # input("Press Enter to move again")
    if ((end_position - current_position).get_length() > (end_position - mid_position).get_length()):
        next_direction = (mid_position - current_position).to_numpy_array()
    else:
        next_direction = (end_position - current_position).to_numpy_array()
    
    next_position = next_direction
    next_magnitude = np.linalg.norm(next_direction)

    # print('current_position: ', current_position.x_val, current_position.y_val, current_position.z_val)
    # print('next_position: ', next_position[0], next_position[1], next_position[2])

    x_next = float(current_position.x_val) +  10 * (float(next_position[0])/next_magnitude)
    y_next = float(current_position.y_val) +  10 * (float(next_position[1])/next_magnitude)
    z_next = float(current_position.z_val)

    # client.simPause(True)
    path = []
    path.append(airsim.Vector3r(x_next, y_next, z_next))
    client.moveOnPathAsync(path, 0.5, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(is_rate=True, yaw_or_rate=1))
    # client.simContinueForTime(5)
    # time.sleep(2)
    # client.simPause(True)

    current_state = client.getMultirotorState()
    current_position = current_state.kinematics_estimated.position
    
    # print("Pausing ..\n")
    # client.simPause(True)

# Loop:
#   * Consider start point, way point, end point
#   * Get current drone position (take ground truth for now, replace with odometry later)
#   * Compute the next position (to simulate next nearest way point from planner)
#   * Command to move it to next way point
#   * Pause the simulation for few seconds

# End motion and recording
print("Done ...")
client.simPause(False)
# client.moveToZAsync(5, 2).join()
# client.moveToPositionAsync(0, 0, 0, 1).join()
client.landAsync(timeout_sec=10).join()
client.armDisarm(False)
client.reset()
client.enableApiControl(False)