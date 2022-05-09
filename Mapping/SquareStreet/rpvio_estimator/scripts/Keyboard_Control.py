import argparse
# from dataclasses import dataclass
import time
import numpy as np

import airsim
import keyboard


# @dataclass
class MotionState:
    vx: int = 0
    vy: int = 0
    vz: int = 0
    yaw_rate: float = 0


def get_args():
    parser = argparse.ArgumentParser("Keyboard AirSim drone runner")
    parser.add_argument('--ip', help='IP of airsim client, leave empty for same machine', default='10.2.36.227')
    parser.add_argument('--duration', help='Duration time for each call', type=int, default=1)
    parser.add_argument('--dv', help='Change in v', type=float, default=3)
    parser.add_argument('--dyaw', help='Change in yaw', type=float, default=np.pi / 4)
    args = parser.parse_args()
    return args


def connect_to_client(ip='') -> airsim.MultirotorClient:
    """
    Connect to client at ip
    :param ip: ip of airsim client
    :return: airsim client
    """
    client = airsim.MultirotorClient(ip=ip)
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    return client


def moveDrone(client: airsim.MultirotorClient, by: MotionState, args):
    """
    Move Drone by given values
    """
    client.moveByVelocityAsync(by.vx, by.vy, by.vz, args.duration, yaw_mode=airsim.YawMode(True, by.yaw_rate))


def detection_loop(client, args):
    """
    Detect keyboard press and move drone accordingly
    :return:
    """
    by = MotionState(0, 0, 0, 0)

    # x
    if keyboard.is_pressed("a"):
        by.vx += args.dv
    if keyboard.is_pressed("d"):
        by.vx -= args.dv

    # y
    if keyboard.is_pressed("w"):
        by.vy += args.dv
    if keyboard.is_pressed("s"):
        by.vy -= args.dv

    # z
    if keyboard.is_pressed("q"):
        by.vz -= args.dv
    if keyboard.is_pressed("e"):
        by.vz += args.dv

    # yaw
    if keyboard.is_pressed("z"):
        by.yaw_rate -= args.dyaw
    if keyboard.is_pressed("c"):
        by.yaw_rate += args.dyaw

    if abs(by.vx) != 0 or abs(by.vy) != 0 or abs(by.vz) != 0 or abs(by.yaw_rate) != 0:
        print(f'vx: {by.vx} vy: {by.vy} vz: {by.vz} yaw_rate:{by.yaw_rate}')
        moveDrone(client, by, args)


def main():
    args = get_args()
    print(args)
    client = connect_to_client(args.ip)
    client.moveToZAsync(-100, 20).join()
    print('Starting listening to keyboard')
    while True:
        detection_loop(client, args)
        time.sleep(1)


if __name__ == '__main__':
    main()