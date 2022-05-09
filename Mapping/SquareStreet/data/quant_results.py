import open3d as o3d
import cv2
import numpy as np

color = cv2.imread('frames/c1_image_1.png', cv2.CV_16U)
depth = cv2.imread('frames/c1_depth_1.png', cv2.CV_16U)

pcd = o3d.geometry.PointCloud()
pts = []

K = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0]
])

inv_K = np.linalg.inv(K)

for i in range(depth.shape[0]):
    for j in range(depth.shape[1]):
        d = depth[i, j]
        if d == 65535:
            continue
        pt = np.array([j, i, 1.0]).reshape((3, 1))
        ray = inv_K.dot(pt)
        ray_direction = ray / np.linalg.norm(ray)
        pt_ = (d/1000.0) * ray_direction
        pts.append(pt_.flatten())

pcd.points = o3d.utility.Vector3dVector(np.array(pts))
o3d.visualization.draw_geometries([pcd])