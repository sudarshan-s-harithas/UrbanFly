import os
import cv2
from options import parse_args
from config import InferenceConfig
import numpy as np
from plane_segmentor import PlaneSegmentor
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
from tqdm import tqdm


def run(options, config, camera):
    n_frames = options.numFrames
    data_dir = options.dataPath
    output_dir = options.dataPath
    segmentor = PlaneSegmentor(options, config, camera)
    for i in tqdm(range(n_frames)):
        rgb_path = os.path.join(data_dir, f'{i}_scene.png')
        print(rgb_path)
        rgb = cv2.imread(rgb_path)
        print(rgb.shape)
        mask_path = os.path.join(data_dir, f'{i}_seg.png')
        print(mask_path)
        building_mask = cv2.imread(mask_path)
        print(building_mask.shape)
        plane_mask = segmentor.segment(rgb, building_mask)
        plane_mask = plane_mask.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f'{i}_plane_mask.png'), plane_mask)


def main():
    args = parse_args()

    if args.dataset == '':
        args.keyname = 'evaluate'
    else:
        args.keyname = args.dataset
        pass
    args.test_dir = 'test/' + args.keyname

    if args.testingIndex >= 0:
        args.debug = True
        pass
    if args.debug:
        args.test_dir += '_debug'
        args.printInfo = True
        pass
    options = args
    config = InferenceConfig(options)
    camera = np.array([320, 320, 320, 240, 640, 480])
    run(options, config, camera)


if __name__ == '__main__':
    main()
