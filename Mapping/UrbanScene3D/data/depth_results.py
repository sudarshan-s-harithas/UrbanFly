import cv2
import numpy as np
import math

color = (0,255,0)
rgb = "%d %d %d" % color

K = np.array([
    [320.0, 0.0, 320.0],
    [0.0, 320.0, 240.0],
    [0.0, 0.0, 1.0]
])

inv_K = np.linalg.inv(K)

def savePointCloud(image, fileName):
  f = open(fileName, "w")
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      pt = image[x,y]
      if (math.isinf(pt[0]) or math.isnan(pt[0])) or (abs(pt[2] - 65535) < 2):
        # skip it
        None
      else: 
        f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2]-1, rgb))
  f.close()

def saveDepthPointCloud(depth_image, fileName):
  f = open(fileName, "w")
  for x in range(depth_image.shape[0]):
    for y in range(depth_image.shape[1]):
      d = depth_image[x,y]
      if d == 65535:
        # skip it
        None
      else: 
        ray = inv_K.dot(np.array([y, x, 1.0]).reshape((3, 1)))
        ray_direction = ray / np.linalg.norm(ray)
        pt = (d/1000.0) * ray_direction
        pt = pt.flatten()
        f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], rgb))
  f.close()

def savePFMPointCloud(pfm_image, fileName):
  f = open(fileName, "w")
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      pt = image[x,y]
      if (math.isinf(pt[0]) or math.isnan(pt[0])) or (abs(pt[2] - 60) < 2):
        # skip it
        None
      else:         
        pt = d * inv_K.dot(np.array([y, x, 1.0]).reshape((3, 1)))
        pt = pt.flatten()
        f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], rgb))
  f.close()

depth_png = cv2.imread('frames/c1_depth_3.png', cv2.CV_16U)
saveDepthPointCloud(depth_png, "pcds/3_depth_pcd.asc")

cv2.imshow("png", depth_png)
cv2.waitKey(0)
cv2.destroyAllWindows()