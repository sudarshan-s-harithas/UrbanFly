import argparse
import glob
import os

import cv2

from optical_flow import OpticalFlowPropagator
import time
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", help="Directory with all the captures", required=True)
    parser.add_argument("--output", "-o", help="Output video name", default="output/")
    parser.add_argument('--fps', default=20)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    files = glob.glob(os.path.join(args.data_dir, "*_scene.png"))
    n_files = len(files)
    h, w, _ = cv2.imread(files[0]).shape
    flow = OpticalFlowPropagator(h, w)
    if os.path.isdir(args.output):
        os.system(f"rm -r {args.output}")

    os.mkdir(args.output)
    os.mkdir(os.path.join(args.output, 'new_captures'))
    writer_rgb = cv2.VideoWriter(os.path.join(args.output, "rgb.mp4"), fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps=args.fps, frameSize=(w, h))
    writer_seg = cv2.VideoWriter(os.path.join(args.output, "seg.mp4"), fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps=args.fps, frameSize=(w, h))
    writer_vis = cv2.VideoWriter(os.path.join(args.output, "vis.mp4"), fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps=args.fps, frameSize=(w, h))

    for i in range(n_files):
        rgb = cv2.imread(os.path.join(args.data_dir, f'{i}_scene.png'))
        seg = None
        if i % 5 == 0:
            seg = cv2.imread(os.path.join(args.data_dir, f'{i}_plane_mask.png'))
            flow.reset(rgb, seg,lk=False)
        else:
            start = time.time()
            seg = flow.propagate_farneback(rgb)
            end = time.time()
            print('time took', end - start)
        vis = cv2.addWeighted(rgb, 1, seg, 0.6, 0)
        writer_vis.write(vis)
        cv2.imwrite(os.path.join(args.output, 'new_captures', f'{i}_plane_mask_flow.png'), seg)

    writer_rgb.release()
    writer_seg.release()
    writer_vis.release()


if __name__ == '__main__':
    main()
