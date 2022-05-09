import numpy as np
import sys


import rospy
import sensor_msgs.msg

sys.path.remove('')
from cv_bridge import CvBridge
import sensor_msgs
from sensor_msgs.msg import Image

from options import parse_args
from config import InferenceConfig
from plane_segmentor import PlaneSegmentor


from rpvio_estimator.srv import PlaneSegmentation, PlaneSegmentationResponse, PlaneSegmentationRequest

segmentor = None
bridge = None


def plane_segmentation_callback(req: PlaneSegmentationRequest):
    global segmentor, bridge
    rospy.loginfo(f"Revice req {req.frame_id}")


def run_server(options, config, camera):
    global segmentor, bridge
#    segmentor = PlaneSegmentor(options, config, camera)
    bridge = CvBridge()
    rospy.init_node('Plane_segmentation_server')
    s = rospy.Service('plane_segmentation', PlaneSegmentation, plane_segmentation_callback)
    print('Server ready!')
    rospy.spin()

    pass


class Node:
    def __init__(self, options, config, camera):
        self.segmentor = PlaneSegmentor(options, config, camera)
        self.bridge = CvBridge()
        pass

    # self.image = None
    # self.rate = rospy.Rate(rate)

    def callback(self, req: PlaneSegmentationRequest):
        rospy.loginfo(f"Received req {req.frame_id}")

        image = self.bridge.imgmsg_to_cv2(req.rgb_image)
        building_mask = self.bridge.imgmsg_to_cv2(req.building_mask_image)
        plane_mask = self.segmentor.segment(image, building_mask)
        print(plane_mask.shape)
        plane_mask = plane_mask.astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(plane_mask,"bgr8")
        msg.header = req.rgb_image.header
        rospy.loginfo(f"Sending back req {req.frame_id}")
        return PlaneSegmentationResponse(req.frame_id, msg)

    def start(self):
        rospy.init_node('plane_segmentation_server')
        s = rospy.Service('plane_segmentation', PlaneSegmentation, self.callback)
        print('Ready for segmentation')
        rospy.spin()


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
    node = Node(options, config, camera)
    node.start()


if __name__ == '__main__':
    main()
