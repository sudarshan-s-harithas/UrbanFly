from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import open3d as o3d
import cvxpy as cp


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
    
    def get_facewise_vertices(self):
        facewise_vertices = {}

        center = self.get_center().reshape((3, 1))
        width = self.dims[0] # x dir
        height = self.dims[1] # y dir
        depth = self.dims[2] # z dir

        # Front face (normal => -ve y-axis)
        front_face_normal = np.array([0, -height/2, 0]).reshape((3, 1))
        front_face_center = center + front_face_normal
        vertex1 = front_face_center + np.array([[width/2, 0, depth/2]]).T
        vertex2 = front_face_center + np.array([[width/2, 0, -depth/2]]).T
        vertex3 = front_face_center + np.array([[-width/2, 0, -depth/2]]).T
        vertex4 = front_face_center + np.array([[-width/2, 0, depth/2]]).T
        front_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['front'] = front_face_vertices

        # Back face (normal => +ve y-axis)
        back_face_normal = np.array([0, height/2, 0]).reshape((3, 1))
        back_face_center = center + back_face_normal
        vertex1 = back_face_center + np.array([[-width/2, 0, depth/2]]).T
        vertex2 = back_face_center + np.array([[-width/2, 0, -depth/2]]).T
        vertex3 = back_face_center + np.array([[width/2, 0, -depth/2]]).T
        vertex4 = back_face_center + np.array([[width/2, 0, depth/2]]).T
        back_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['back'] = back_face_vertices

        # Right face (normal  => +ve x-axis)
        right_face_normal = np.array([width/2, 0, 0]).reshape((3, 1))
        right_face_center = center + right_face_normal
        vertex1 = right_face_center + np.array([[0, height/2, depth/2]]).T
        vertex2 = right_face_center + np.array([[0, height/2, -depth/2]]).T
        vertex3 = right_face_center + np.array([[0, -height/2, -depth/2]]).T
        vertex4 = right_face_center + np.array([[0, -height/2, depth/2]]).T
        right_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['right'] = right_face_vertices

        # Left face (normal => -ve x-axis)
        left_face_normal = np.array([-width/2, 0, 0]).reshape((3, 1))
        left_face_center = center + left_face_normal
        vertex1 = left_face_center + np.array([[0, -height/2, depth/2]]).T
        vertex2 = left_face_center + np.array([[0, -height/2, -depth/2]]).T
        vertex3 = left_face_center + np.array([[0, height/2, -depth/2]]).T
        vertex4 = left_face_center + np.array([[0, height/2, depth/2]]).T
        left_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['left'] = left_face_vertices

        # Top face (normal => +ve z-axis)
        top_face_normal = np.array([0, 0, depth/2]).reshape((3, 1))
        top_face_center = center + top_face_normal
        vertex1 = top_face_center + np.array([[width/2, height/2,  0]]).T
        vertex2 = top_face_center + np.array([[width/2, -height/2,  0]]).T
        vertex3 = top_face_center + np.array([[-width/2, -height/2, 0]]).T
        vertex4 = top_face_center + np.array([[-width/2, height/2, 0]]).T
        top_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['top'] = top_face_vertices

        # Bottom face (normal => -ve z-axis)
        bottom_face_normal = np.array([0, 0, -depth/2]).reshape((3, 1))
        bottom_face_center = center + bottom_face_normal
        vertex1 = bottom_face_center + np.array([[width/2, -height/2,  0]]).T
        vertex2 = bottom_face_center + np.array([[width/2, height/2,  0]]).T
        vertex3 = bottom_face_center + np.array([[-width/2, height/2, 0]]).T
        vertex4 = bottom_face_center + np.array([[-width/2, -height/2, 0]]).T
        bottom_face_vertices = np.vstack((vertex1.T, vertex2.T, vertex3.T, vertex4.T))
        facewise_vertices['bottom'] = bottom_face_vertices

        return facewise_vertices

    def get_plane_params(self, vertices):
        params = []

        vertex1 = vertices[0:1, :].T
        vertex2 = vertices[1:2, :].T
        vertex3 = vertices[2:3, :].T
        vertex4 = vertices[3:4, :].T

        # Compute normal
        vertical_dir = vertex1 - vertex2
        vertical_dir /= np.linalg.norm(vertical_dir)

        horizontal_dir = vertex3 - vertex2
        horizontal_dir /= np.linalg.norm(horizontal_dir)
        
        normal = np.cross(vertical_dir.flatten(), horizontal_dir.flatten()).reshape((3, 1))
        normal /= np.linalg.norm(normal)

        d = -np.dot(normal.T, vertex4)

        params.append(normal[0, 0])
        params.append(normal[1, 0])
        params.append(normal[2, 0])
        params.append(d)

        return np.array(params).reshape((4, 1))

    def get_face_planes(self):
        face_planes = []

        return face_planes

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

        if min_sdf <= 2:
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

num_goal = 3000 ####### batch size
num = 80

x_init =  30.0
y_init =  -5.0
z_init =  0.0
yaw_init = 0.0

x_des_traj_init = x_init
y_des_traj_init = y_init
z_des_traj_init = z_init
yaw_des_traj_init = yaw_init

vx_des = 1.0
vy_des = -0.70
vz_des = -0.2
vyaw_des = -np.pi/200

############################################################## Hyperparameters
t_fin = 75

######################################## noise sampling

########### Random samples for batch initialization of heading angles
A = np.diff(np.diff(np.identity(num), axis = 0), axis = 0)
# A = np.diff(np.identity(prob.num), axis =0 )
# print(A.shape)
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

x_fin = x_des_traj_init+vx_des*t_fin
y_fin = y_des_traj_init+vy_des*t_fin
z_fin = z_des_traj_init+vz_des*t_fin
yaw_fin = yaw_des_traj_init+vyaw_des*t_fin

t_interp = np.linspace(0, t_fin, num)
x_interp = x_des_traj_init + ((x_fin-x_des_traj_init)/t_fin) * t_interp
y_interp = y_des_traj_init + ((y_fin-y_des_traj_init)/t_fin) * t_interp
z_interp = z_des_traj_init + ((z_fin-z_des_traj_init)/t_fin) * t_interp
yaw_interp = yaw_des_traj_init + ((yaw_fin-yaw_des_traj_init)/t_fin) * t_interp

# A_mat = A
# print(temp_1.shape)
# print(A_mat.shape)
R = np.dot(A_mat.T, A_mat)
mu = np.zeros(num)
cov = np.linalg.pinv(R)

# print(R.shape)
################# Gaussian Trajectory Sampling
eps_kx = np.random.multivariate_normal(mu, 0.03*cov, (num_goal, ))
eps_ky = np.random.multivariate_normal(mu, 0.03*cov, (num_goal, ))
eps_kz = np.random.multivariate_normal(mu, 0.03*cov, (num_goal, ))
eps_kyaw = np.random.multivariate_normal(mu, 0.01*cov, (num_goal, ))

x_samples = x_interp+eps_kx
y_samples = y_interp+eps_ky
z_samples = z_interp+eps_kz
yaw_samples = yaw_interp+0.0*eps_kyaw

print(eps_kx.shape)
print(x_samples.shape)
print(y_samples.shape)
print(z_samples.shape)
print(yaw_samples.shape)

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

box = o3d.geometry.TriangleMesh.create_box(width=10.0, height=40.0, depth=50.0)
box.translate(np.array([40, -15, -40]).reshape((3, 1)))

"""
Visualize the sample trajectories in matplotlib 3d
"""
fig = plt.figure()
ax = plt.axes(projection='3d')

traj_points = []
traj_colors = []
traj_frames = []

both = True

for x_s, y_s, z_s, yaw_s in zip(x_samples, y_samples, z_samples, yaw_samples):
    # ax.plot3D(x_s.T, y_s.T, z_s.T, 'blue', linewidth=0.1)
    
    is_colliding = bworld2.is_colliding_trajectory(x_s, y_s, z_s)

    if (not is_colliding) or both:
        if not is_colliding:
            colors = [[0, 0, 1] for i in range(len(x_s))]
        else:
            colors = [[1, 0, 0] for i in range(len(x_s))]

        pts = []
        for x, y, z, yaw in zip(x_s, y_s, z_s, yaw_s):
            # t_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # R = np.eye(3)
            # R[:2, :2] = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # t_frame.rotate(R)
            # t_frame.translate([x, y, z])
            # traj_frames += [t_frame]

            pts += [[x, y, z]]
        
        traj_points += pts
        traj_colors += colors

# plt.show()

coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
coord_frame.translate(np.array([40, 0, 0]).reshape((3, 1)))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(traj_points))
pcd.colors = o3d.utility.Vector3dVector(np.array(traj_colors))

o3d.visualization.draw_geometries([pcd, coord_frame, line_set] + bworld2.geometries + traj_frames)

trajectory1 = np.array(traj_points)[:80, :3]

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(trajectory1)
# pcd2.colors = o3d.utility.Vector3dVector(np.array(traj_colors))

o3d.visualization.draw_geometries([pcd2, coord_frame, line_set] + bworld2.geometries)

x = cp.Variable((80, 3))
x.value = trajectory1
constraint = [x[0, 0] == trajectory1[0, 0], x[0, 1] == trajectory1[0, 1], x[0, 2] == trajectory1[0, 2]]
constraint += [x[79, 0] == trajectory1[79, 0], x[79, 1] == trajectory1[79, 1], x[79, 2] == trajectory1[79, 2]]
vertices = bworld2.boxes[0].get_facewise_vertices()
right_plane = bworld2.boxes[0].get_plane_params(vertices['front'])
r_plane = right_plane[:3, :]/right_plane[3, 0]

print("plane params : ", r_plane.shape)

good_points = []
good_colors = []
for point in trajectory1:
    point_ = np.array([point[0], point[1], point[2], 1.0]).reshape((4, 1))
    if point_.T @ right_plane > 0:
        good_points.append([point[0], point[1], point[2]])
        good_colors.append([0, 0, 0])

pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(np.array(good_points))
pcd3.colors = o3d.utility.Vector3dVector(np.array(good_colors))

# plane = cp.Parameter((3, 1))
# plane.value = r_plane.reshape((3, 1))

onz = cp.Parameter((80, 1))
onz.value = np.ones((80, 1))
# constraint += [x @ r_plane - 20*onz >= 0]

obj = cp.Minimize(cp.sum_squares((x[1:,:]-x[:79,:])))
problem = cp.Problem(obj, constraint)
problem.solve()
print('Solving ...: ', problem.solve())
print('Status of the problem: ', problem.status)
# print('x value is ', x.value)

traj_opt = x.value
pcd_opt = o3d.geometry.PointCloud()
pcd_opt.points = o3d.utility.Vector3dVector(traj_opt)
# pcd2.colors = o3d.utility.Vector3dVector(np.array(traj_colors))

o3d.visualization.draw_geometries([pcd2, pcd3, coord_frame, line_set] + bworld2.geometries)

# # Sample many points around the boxes

# # Determine if the points are inside or outside the plane

# test_points = []
# test_colors = []
# for x in range(30, 120, 5):
#     for y in range(-60, 10, 5):
#         point_ = np.array([[x, y, -5, 1]])

#         vertices = bworld2.boxes[0].get_facewise_vertices()
#         right_plane = bworld2.boxes[0].get_plane_params(vertices['front'])

#         test_points.append([x, y, -5])

#         if point_ @ right_plane >= 0:
#             test_colors.append([1, 0, 0])
#         else:
#             test_colors.append([0, 0, 0])

# test_pcd = o3d.geometry.PointCloud()
# test_pcd.points = o3d.utility.Vector3dVector(np.array(test_points))
# test_pcd.colors = o3d.utility.Vector3dVector(np.array(test_colors))

# o3d.visualization.draw_geometries([test_pcd] + bworld2.geometries)

# test_points = []
# test_colors = []
# for x in range(30, 120, 5):
#     for y in range(-60, 10, 5):
#         point_ = np.array([[x, y, -5, 1]])

#         vertices = bworld2.boxes[0].get_facewise_vertices()
#         right_plane = bworld2.boxes[0].get_plane_params(vertices['right'])

#         test_points.append([x, y, -5])

#         if point_ @ right_plane >= 0:
#             test_colors.append([1, 0, 0])
#         else:
#             test_colors.append([0, 0, 0])

# test_pcd = o3d.geometry.PointCloud()
# test_pcd.points = o3d.utility.Vector3dVector(np.array(test_points))
# test_pcd.colors = o3d.utility.Vector3dVector(np.array(test_colors))

# o3d.visualization.draw_geometries([test_pcd] + bworld2.geometries)
# test_points = []
# test_colors = []
# for x in range(30, 120, 5):
#     for y in range(-60, 10, 5):
#         point_ = np.array([[x, y, -5, 1]])

#         vertices = bworld2.boxes[0].get_facewise_vertices()
#         right_plane = bworld2.boxes[0].get_plane_params(vertices['back'])

#         test_points.append([x, y, -5])

#         if point_ @ right_plane >= 0:
#             test_colors.append([1, 0, 0])
#         else:
#             test_colors.append([0, 0, 0])

# test_pcd = o3d.geometry.PointCloud()
# test_pcd.points = o3d.utility.Vector3dVector(np.array(test_points))
# test_pcd.colors = o3d.utility.Vector3dVector(np.array(test_colors))

# o3d.visualization.draw_geometries([test_pcd] + bworld2.geometries)
# test_points = []
# test_colors = []
# for x in range(30, 120, 5):
#     for y in range(-60, 10, 5):
#         point_ = np.array([[x, y, -5, 1]])

#         vertices = bworld2.boxes[0].get_facewise_vertices()
#         right_plane = bworld2.boxes[0].get_plane_params(vertices['left'])

#         test_points.append([x, y, -5])

#         if point_ @ right_plane >= 0:
#             test_colors.append([1, 0, 0])
#         else:
#             test_colors.append([0, 0, 0])

# test_pcd = o3d.geometry.PointCloud()
# test_pcd.points = o3d.utility.Vector3dVector(np.array(test_points))
# test_pcd.colors = o3d.utility.Vector3dVector(np.array(test_colors))

# o3d.visualization.draw_geometries([test_pcd] + bworld2.geometries)