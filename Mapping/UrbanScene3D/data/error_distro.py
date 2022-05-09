import matplotlib.pyplot as plt
import numpy as np
import math

error_data = np.loadtxt('minihattan/error_params.txt')
error_data = error_data[error_data[:, 0]<5]
error_data = error_data[np.argsort(error_data[:, 1])] #sort error based on the viewing angle
all_angles = error_data[:, 1] * 180 / math.pi

view_angle = all_angles[all_angles<=90]
x = error_data[all_angles<=90][:, 0]
# hist, bins = np.histogram(view_angle, bins=200, density=True)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.xlabel('error of (d) in meters')
# plt.ylabel('probability of error')
# plt.title('Probability distribution of plane depth error w.r.t ground truth planes')
# plt.show()

# plt.plot(view_angle, x)
# plt.show()

# colors = view_angle
# fig = plt.figure()
# ax = fig.add_subplot(projection='polar')
# c = ax.scatter(view_angle, x, c=colors, cmap='hsv', alpha=0.75)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
hist, xedges, yedges = np.histogram2d(view_angle, x, bins=100, range=[[0, 90], [0, 5]])

# Construct arrays for the anchor positions of the 16 bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = np.ones_like(zpos)
dy = 0.1 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()