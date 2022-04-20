import open3d as o3d
import cv2
import numpy as np

K = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0]
])

inv_K = np.linalg.inv(K)

depth = cv2.imread('frames/c1_depth_1.png', cv2.CV_16U)
mask = cv2.imread('frames/c1_mask_1.png', -1)

color_map = {}
color_map[0] = 0

def rgb2code(r, g, b):
    hex_rgb = ((r & 0xff) << 16) + ((g & 0xff) << 8) + (b & 0xff)
    
    if ((r == 7) and (g == 93) and (b == 182)):
        return rgb2code(0, 0, 0)

    if hex_rgb not in color_map:
        color_map[hex_rgb] = len(color_map)

    return color_map[hex_rgb]

id_mask = np.zeros((mask.shape[0], mask.shape[1]))

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        r, g, b = mask[i, j]
        id_mask[i, j] = rgb2code(r, g, b)

params_map = []

for hid in color_map:
    pid = color_map[hid]
    if pid == 0:
        continue
    xs, ys = np.where(id_mask == pid)

    if len(xs) < 50:
        continue

    zs = depth[xs, ys]/1000.0

    pix = np.vstack((ys, xs, np.ones(xs.shape)))
    
    rays = inv_K.dot(pix)
    rays_mags = np.linalg.norm(rays, axis=0)
    rays_dirs = rays/rays_mags

    pts = zs * rays_dirs

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts.T)
    # o3d.visualization.draw_geometries([pcd])
    
    pts_ = np.vstack((pts, np.ones((1, pts.shape[1])))).astype(np.float32)
    _, _, Vt = np.linalg.svd(pts_.T, full_matrices=False)
    params = Vt.T[:, -1]
    params = params/np.linalg.norm(params[:3])
    print("pid: "+str(pid)+"; "+str(xs.shape[0])+" points")
    print(params)
    params_map.append([pid, params[0], params[1], params[2], params[3]])

np.savetxt('c1_planes1.txt', np.array(params_map), fmt='%g')