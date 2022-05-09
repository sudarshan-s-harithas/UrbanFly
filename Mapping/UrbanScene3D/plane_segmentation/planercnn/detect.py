import numpy as np
import cv2
import torch

from utils import *
from datasets.plane_dataset import *
from evaluate import PlaneRCNNDetector


#
class Detector:
    def __init__(self, options, config, camera):
        camera = np.array(camera)
        # camera[[0, 2, 4]] *= 640.0 / camera[4]
        # camera[[1, 3, 5]] *= 480.0 / camera[5]
        self.camera = camera
        self.config = config
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                config.BACKBONE_SHAPES,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)

        self.detector = PlaneRCNNDetector(options, config, modelType='final')

        self.preprocess_init()

    def preprocess_init(self):
        self.depth = np.zeros(
            (self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM),
            dtype=np.float32)
        segmentation = np.zeros(
            (self.config.IMAGE_MIN_DIM, self.config.IMAGE_MAX_DIM),
            dtype=np.int32)

        planes = np.zeros((1, 3))

        instance_masks = []
        class_ids = []
        parameters = []

        if len(planes) > 0:
            plane_offsets = np.linalg.norm(planes, axis=-1)
            plane_normals = planes / np.maximum(
                np.expand_dims(plane_offsets, axis=-1), 1e-4)
            distances_N = np.linalg.norm(np.expand_dims(plane_normals,
                                                        1) - self.config.ANCHOR_NORMALS,
                                         axis=-1)
            normal_anchors = distances_N.argmin(-1)

        for planeIndex, plane in enumerate(planes):
            m = segmentation == planeIndex
            if m.sum() < 1:
                continue
            instance_masks.append(m)
            class_ids.append(normal_anchors[planeIndex] + 1)
            normal = plane_normals[planeIndex] - self.config.ANCHOR_NORMALS[
                normal_anchors[planeIndex]]
            parameters.append(np.concatenate([normal, np.zeros(1)], axis=0))

        self.parameters = np.array(parameters)
        self.mask = np.stack(instance_masks, axis=2)
        self.class_ids = np.array(class_ids, dtype=np.int32)
        self.segmentation = segmentation
        self.planes = planes

    def preprocess(self, image):
        extrinsics = np.eye(4, dtype=np.float32)
        camera = self.camera
        # don't need this as image is already 640 x 480
        # image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_LINEAR)

        image, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_parameters = load_image_gt(
            self.config, 0, image, self.depth, self.mask, self.class_ids,
            self.parameters,
            augment=False)
        ## RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes,
                                                self.config)

        ## If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES,
                replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]
            gt_parameters = gt_parameters[ids]
            pass

        ## Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        image = utils.mold_image(image.astype(np.float32), self.config)

        depth = np.concatenate(
            [np.zeros((80, 640)), self.depth, np.zeros((80, 640))],
            axis=0).astype(
            np.float32)
        segmentation = np.concatenate(
            [np.full((80, 640), fill_value=-1), self.segmentation,
             np.full((80, 640), fill_value=-1)], axis=0).astype(np.float32)

        data_pair = [image.transpose((2, 0, 1)).astype(np.float32), image_metas,
                     rpn_match.astype(np.int32), rpn_bbox.astype(np.float32),
                     gt_class_ids.astype(np.int32), gt_boxes.astype(np.float32),
                     gt_masks.transpose((2, 0, 1)).astype(np.float32),
                     gt_parameters[:, :-1].astype(np.float32),
                     depth.astype(np.float32), extrinsics.astype(np.float32),
                     self.planes.astype(np.float32),
                     segmentation.astype(np.int64),
                     gt_parameters[:, -1].astype(np.int32)]
        data_pair = data_pair + data_pair

        data_pair.append(np.zeros(7, np.float32))

        data_pair.append(self.planes)
        data_pair.append(self.planes)
        data_pair.append(np.zeros((len(self.planes), len(self.planes))))
        data_pair.append(camera.astype(np.float32))
        return data_pair

    def run(self, image):
        """
        Run the detector on image
        :param image:
        :return: (masks, parameters)
        """
        sample = self.preprocess(image)
        sample = [torch.tensor(data[None, ...]) for data in sample]
        with torch.no_grad():
            detection_pair = self.detector.detect(sample)
        parameters = detection_pair[0]['detection'][:,
                     6:9].detach().cpu().numpy()
        masks = detection_pair[0]['masks'][:, 80:560].detach().cpu().numpy()

        return masks, parameters
