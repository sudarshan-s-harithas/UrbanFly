from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

"""Utilies to create a box world"""

class Box:
    dims = np.ones((3, 1)) # [width(x), height(y), depth(z)]
    mesh = None

    def __init__(self, dims_):
        self.dims = dims_
        self.mesh = o3d.geometry.TriangleMesh.create_box(width=dims_[0], height=dims_[1], depth=dims_[2])

    def get_center(self):
        return self.mesh.get_center()


    def get_sdf(self, wp, center_noise=np.zeros((3,1)), dims_noise=np.zeros((3,1))):
        p = wp - (self.get_center() + center_noise) # transform point to box coordinate frame
        d = (np.array(self.dims) + dims_noise)/2 # Only along half dims

        q = abs(p) - d

        sdf = np.linalg.norm(np.clip(q, 0, None)) + min(q.max(), 0)

        return sdf

class BoxWorld:
    boxes = []
    gt_sdfs = None
    geometries = None

    def show(self):
        if self.geometries is None:
            self.geometries = []

            # Add a coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.geometries.append(coord_frame)

            # Add boxes
            self.geometries += [box.mesh for box in self.boxes]

        o3d.visualization.draw_geometries(self.geometries)

    def create_boxes(self, dims, translations):
        # Construct all boxes
        for dim, translation in zip(dims, translations):
            box = Box(dim.flatten())
            box.mesh.translate(translation)

            self.boxes.append(box)

    def is_point_inside(self, point):
        min_sdf = 10000
        for box in self.boxes:
            min_sdf = min(box.get_sdf(point), min_sdf)

        if min_sdf <= 1e-2:
            return True
        else:
            return False


    def is_colliding_trajectory(self, x_s, y_s, z_s):
        pts = np.vstack((x_s.flatten(), y_s.flatten(), z_s.flatten())).T
        
        for pt in pts:
            if self.is_point_inside(pt):
                return True

        return False

# Create a world
# bworld = BoxWorld()

# Add boxes
dims = [
    [1, 1, 6],
    [1, 1, 7],
    [1, 1, 6],
    [1, 1, 7],
    [1, 1, 8],
    [1, 1, 6],
    [1, 1, 7],
    [1, 1, 8],
    [1, 1, 6],
    [1, 1, 7],
    [1, 1, 6],
    [1, 1, 8],
    [1, 1, 6],
    [1, 1, 7],
    [1, 1, 5],
    [1, 1, 4],
    [1, 1, 5],
    [1, 1, 4]
]
translations = [
    [1, 1, 0],
    [3, 1, 0],
    [5, 1, 0],
    [7, 1, 0],
    [9, 1, 0],
    [1, 7, 0],
    [3, 7, 0],
    [5, 7, 0],
    [7, 7, 0],
    [9, 7, 0],
    [1, 3, 0],
    [1, 5, 0],
    [9, 3, 0],
    [9, 5, 0],
    [3.5, 3, 0],
    [3.5, 5, 0],
    [5.5, 3, 0],
    [5.5, 5, 0]
]
# bworld.create_boxes(np.array(dims), np.array(translations))

# Show the world
# bworld.show()

"""
Sample many trajectories in 3D between start and end points
Visualize those trajectories in open3d
"""

num_goal = 1000 ####### batch size
num = 70

x_init =  20.0
y_init =  -10.0
z_init =  0.0

x_des_traj_init = x_init
y_des_traj_init = y_init
z_des_traj_init = z_init

vx_des = 1.0
vy_des = 0.0
vz_des = 0.0

############################################################## Hyperparameters
t_fin = 100

######################################## noise sampling

########### Random samples for batch initialization of heading angles
A = np.diff(np.diff(np.identity(num), axis = 0), axis = 0)
# A = np.diff(np.identity(prob.num), axis =0 )

temp_1 = np.zeros(num)
temp_2 = np.zeros(num)
temp_3 = np.zeros(num)
temp_4 = np.zeros(num)

temp_1[0] = 1.0
temp_2[0] = -2
temp_2[1] = 1
temp_3[-1] = -2
temp_3[-2] = 1

temp_4[-1] = 1.0

A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
# A_mat = A

R = np.dot(A_mat.T, A_mat)
mu = np.zeros(num)
cov = np.linalg.pinv(R)

################# Gaussian Trajectory Sampling
eps_k = np.random.multivariate_normal(mu, 0.03*cov, (num_goal, ))
eps_kz = np.random.multivariate_normal(mu, 0.03*cov, (num_goal, ))

x_fin = x_des_traj_init+vx_des*t_fin
y_fin = y_des_traj_init+vy_des*t_fin
z_fin = z_des_traj_init+vz_des*t_fin

x_interp = np.linspace(x_des_traj_init, x_fin, num)
y_interp = np.linspace(y_des_traj_init, y_fin, num)
z_interp = np.linspace(z_des_traj_init, z_fin, num)

x_samples = x_interp+0.0*eps_k 
y_samples = y_interp+eps_k
z_samples = z_interp+eps_kz
# z_samples = z_interp+0.0*eps_kz

print(eps_k.shape)
print(x_samples.shape)
print(y_samples.shape)
print(z_samples.shape)

""" Create a box world"""
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([
        [x_init, y_init, z_init],
        [x_fin, y_fin, z_fin]
    ]),
    lines=o3d.utility.Vector2iVector([
        [0, 1]
    ]),
)

bworld2 = BoxWorld()
dims2 = [
    [30, 25, 50],
    [40, 20, 50]
]
translations2 = [
    [50, -20, -40],
    [50, -50, -40],

]
bworld2.create_boxes(np.array(dims2), np.array(translations2))
bworld2.show()

# box = o3d.geometry.TriangleMesh.create_box(width=10.0, height=40.0, depth=50.0)
# box.translate(np.array([40, -15, -40]).reshape((3, 1)))

"""
Visualize the sample trajectories in matplotlib 3d
"""
fig = plt.figure()
ax = plt.axes(projection='3d')

traj_points = []
traj_colors = []

for x_s, y_s, z_s in zip(x_samples, y_samples, z_samples):
    ax.plot3D(x_s.T, y_s.T, z_s.T, 'blue', linewidth=0.1)
    pts = [[x, y, z] for x, y, z in zip(x_s, y_s, z_s)]
    
    if bworld2.is_colliding_trajectory(x_s, y_s, z_s):
        colors = [[1, 0, 0] for i in range(len(x_s))]
    else:
        colors = [[0, 0, 1] for i in range(len(x_s))]

    traj_points += pts
    traj_colors += colors

plt.show()

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_frame.translate(np.array([40, 0, 0]).reshape((3, 1)))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(traj_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(traj_colors))

o3d.visualization.draw_geometries([pcd, coord_frame, line_set] + bworld2.geometries)