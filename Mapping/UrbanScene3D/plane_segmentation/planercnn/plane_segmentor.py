import cv2
import numpy as np
from detect import Detector


#
class PlaneSegmentor:
    def __init__(self, options, config, camera, ids_file='seg_rgbs.txt',
                 ground_id=212, sky_id=24, min_size_thresh=8000):
        self.camera = camera
        self.config = config
        self.options = options
        self.detector = Detector(options, config, camera)
        self.mapping = {}
        self.rev_mapping = -1 * np.ones((256, 256, 256), dtype=int)
        self.save_ids(ids_file)
        self.ground_id = ground_id
        self.sky_id = sky_id
        self.thresh = min_size_thresh

    def save_ids(self, ids_file: str):
        with open(ids_file, 'r') as f:
            for line in f.readlines():
                _id, color = tuple(line[:-1].split('\t'))
                _id = int(_id)
                color = np.fromstring(color[1:-1], dtype=np.uint8, sep=',')
                self.mapping[_id] = color
                self.rev_mapping[color[0], color[1], color[2]] = _id

    def segment(self, rgb_image, seg_image) -> np.ndarray:
        """
        Run
        - Plane detector
        - Plane filtering
        - Mask merging

        Returns: plane segmented image
        """
        masks, parameters = self.detect_planes(rgb_image)
        masks = self.filter_planes(rgb_image, seg_image, masks)
        plane_mask = self.merge_masks(masks)

        return plane_mask

    def detect_planes(self, rgb_image) -> np.ndarray:
        """
        Detect planes
        rgb_image: h x w x 3 rgb image

        Return: masks, parameters
        """
        return self.detector.run(rgb_image)

    def filter_planes(self, rgb: np.ndarray, building_seg: np.ndarray,
                      masks: np.ndarray) -> np.ndarray:
        """
        Filter masks

        Returns: filtered masks
        """

        # image is in BGR, ids are for RGB
        id_image = self.rev_mapping[
            building_seg[:, :, 2], building_seg[:, :, 1], building_seg[:, :, 0]]
        sky_mask = id_image == self.sky_id
        ground_mask = id_image == self.ground_id

        for i in range(masks.shape[0]):
            masks[i][sky_mask | ground_mask] = 0
            if (masks[i] == 1).sum() < self.thresh:
                masks[i] = 0

            grayscale = masks[i] * 255
            res = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            res = cv2.morphologyEx(res, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            res = cv2.erode(res, np.ones((5, 5), np.uint8), iterations=2)
            masks[i] = (res == 255).astype(np.uint8)

        return masks

    def merge_masks(self, masks) -> np.ndarray:
        """
        Merge n masks into a segmented image
        """
        plane_seg = np.zeros((*masks.shape[1:], 3))

        for i, mask in enumerate(masks):
            plane_seg[mask == 1] = self.mapping[i + 1]

        return plane_seg
