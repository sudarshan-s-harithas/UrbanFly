import cv2
import numpy as np
from scipy.ndimage import shift
import time

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(
                     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))


def segregate_plane_mask(plane_mask: np.ndarray):
    """
    Segregate the mask into its individual components
    :param plane_mask:
    :return:
    """
    u, indices = np.unique(plane_mask.reshape(-1, 3), axis=0, return_inverse=True)
    id_image = indices.reshape(plane_mask.shape[:2])
    masks = np.zeros((u.shape[0], *plane_mask.shape[:2]))

    for i in range(masks.shape[0]):
        masks[i][id_image == i] = 1

    assert np.all(u[0] == 0), "ERROR!!"

    return masks, id_image, u


class OpticalFlowPropagator:
    def __init__(self, h, w):
        self.masks: np.ndarray = np.array([])
        self.old: np.ndarray = np.array([])
        self.p0: np.ndarray = np.array([])
        self.id_mask: np.ndarray = np.array([])
        self.old_gray: np.ndarray = np.array([])
        self.plane_mask: np.ndarray = np.array([])
        self.i_dx = None
        self.i_n = None
        self.mapping = None
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        self.coords = np.float32(np.dstack([x_coords, y_coords]))

    def reset(self, frame: np.ndarray, plane_mask: np.ndarray, lk=False):
        self.masks, self.id_mask, self.mapping = segregate_plane_mask(plane_mask)
        self.old = frame
        self.old_gray = cv2.cvtColor(self.old, cv2.COLOR_BGR2GRAY)
        if lk:
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None,
                                              **feature_params)
        self.i_dx = np.zeros((self.masks.shape[0], 2))
        self.i_n = np.zeros((self.masks.shape[0],), dtype=int)
        self.plane_mask = plane_mask

    def merge_masks(self, masks):
        plane_seg = np.zeros((*masks.shape[1:], 3))

        for i, mask in enumerate(masks):
            plane_seg[mask == 1] = self.mapping[i]

        return plane_seg

    def propagate_none(self, frame):
        return self.plane_mask

    def propagate_lk(self, frame: np.ndarray):
        """
        Propagate the mask to current frame
        :param frame: new frame
        :return: Propagated plane mask
        """
        old_gray = cv2.cvtColor(self.old, cv2.COLOR_RGB2GRAY)
        assert len(self.p0) and len(old_gray), "features not known, please reset"
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                               self.p0, None, **lk_params)
        good_new = good_old = None
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

        # now we have good_new and good_old features, get the change and propagate the mask accordingly

        self.i_dx = np.zeros((self.masks.shape[0], 2))
        self.i_n = np.zeros((self.masks.shape[0],), dtype=int)

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            new = new.ravel()
            old = old.ravel()
            old = np.clip(old, 0, np.array(self.id_mask.shape[:2])[::-1] - 1)
            _id = self.id_mask[int(old[1]), int(old[0])]
            dx = new - old

            # NOTE: If we need to do iteratively
            self.i_dx[_id] += dx
            self.i_n[_id] += 1

        available_masks = self.masks[self.i_n > 0]
        i_dx = self.i_dx[self.i_n > 0].reshape(-1, 2)
        i_n = self.i_n[self.i_n > 0].reshape(-1, 1)
        i_dx /= i_n
        new_masks = np.zeros_like(available_masks)

        # print(i_dx[:,0].max(), i_dx[:,0].min())

        # self.id_mask = np.zeros_like(self.id_mask)
        for i in range(available_masks.shape[0]):
            # shift_vertically = i_dx[i][0]
            # if shift_vertically >= 0:
            #     shift_horizontally = i_dx[i][1]
            #     if shift_horizontally >= 0:
            #         new_masks[i][shift_vertically:, shift_horizontally:] = available_masks[i][:-shift_vertically,
            #                                                                :-shift_horizontally]
            #     else:
            #         new_masks[i][shift_vertically:, : shift_horizontally] = available_masks[i][:-shift_vertically,
            #                                                                 -shift_horizontally:]
            # else:
            #     shift_horizontally = i_dx[i][1]
            #     if shift_horizontally >= 0:
            #         new_masks[i][:shift_vertically, shift_horizontally:] = available_masks[i][-shift_vertically:,
            #                                                                :-shift_horizontally]
            #     else:
            #         new_masks[i][:shift_vertically, : shift_horizontally] = available_masks[i][-shift_vertically:,
            #                                                                 -shift_horizontally:]
            difference = np.round(i_dx[i]).astype('int')
            new_masks[i] = shift(available_masks[i], (difference[0], difference[1]))
            dir = ["None", "None"]
            if difference[0] > 0:
                dir[0] = "right"
            elif difference[0] < 0:
                dir[0] = "left"

            if difference[1] > 0:
                dir[1] = "down"
            elif difference[1] < 0:
                dir[1] = "up"
            print(dir)
            # self.id_mask[new_masks[i] == 1] = i

        # re-init all params (for iterative updating)

        # self.masks = new_masks
        # self.old_gray = frame_gray

        # NOTE: If we need regular updating
        # self.p0 = good_new.reshape(-1, 1, 2)

        mask = self.merge_masks(self.masks).astype(np.uint8)

        # for i, (new, old) in enumerate(zip(good_new, good_old)):
        #     a, b = new.ravel()
        #     c, d = old.ravel()
        #
        #     mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        #     mask = cv2.circle(mask, (int(a), int(b)), 5, color[i].tolist(), -1)

        return mask

    def propagate_farneback(self, frame: np.ndarray):
        # return self.plane_mask
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, .5, 3, 15, 3, 5, 1.2, 0)

        pixel_map = self.coords - flow
        new_mask = cv2.remap(self.plane_mask, pixel_map, None, cv2.INTER_LINEAR)

        self.plane_mask = new_mask
        self.old_gray = frame_gray
        self.old = frame

        return new_mask

# TODO can also iteratively
