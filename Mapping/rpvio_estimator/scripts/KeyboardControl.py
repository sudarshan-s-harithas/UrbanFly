import numpy as np
import tty, sys, termios
import airsim
import rospy
from itertools import count

filedescriptors = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin)
x = 0

Done = False

Xpose =0
Ypose =0
Zpose =0 
client = airsim.MultirotorClient(ip='10.2.36.227')
# client.reset()
client.confirmConnection()
client.enableApiControl(True)
# client.takeoffAsync()
# client.hoverAsync()

# client.moveToZAsync(-2, 0.5)
# client.moveToZAsync(0, 0.5)
# client.moveToZAsync(-2.5, 0.5)

# pt1 = [ 0.0, 0.0, -3 ]
# pt2 = [0.0, 0.0, -4  ]

# client.moveToPositionAsync( pt1[0] , pt1[1] , pt1[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0)).join()
# client.moveToPositionAsync( pt2[0] , pt2[1] , pt2[2] , 0.25, np.inf, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(True, 0.0)).join()



vx , vy, vz = 1,1,1 
duration = 0.5

def GetCommand( val ):

	if( val == "8" ):
		client.moveByVelocityAsync( 0, -vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))
		print("in")

	if( val == "5" ):
		client.moveByVelocityAsync( 0, 0, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))

	if( val == "2"):

		client.moveByVelocityAsync( 0, vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))

	if(val == "4"):
		client.moveByVelocityAsync( -vy, 0, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))
	if(val == "6"):
		client.moveByVelocityAsync( vy, 0, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))


	if(val == "7" ):
		client.moveByVelocityAsync( -vy, -vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))
	if(val == "9"):
		client.moveByVelocityAsync( vy, -vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))

	if(val == "1" ):
		client.moveByVelocityAsync( -vy, vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))
	if(val == "3"):
		client.moveByVelocityAsync( vy, vx, 0, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))
	if(val == "w"):
		client.moveByVelocityAsync( 0, 0, vz, duration, yaw_mode=airsim.YawMode(True, np.pi / 4))

P =100
for i in count(0):
  x=sys.stdin.read(1)[0]
  print("You pressed", x)

  P += 1
  GetCommand( x )
  if x == "x":
    break
termios.tcsetattr(sys.stdin, termios.TCSADRAIN,filedescriptors)


	




if __name__ == '__main__':
    listener()