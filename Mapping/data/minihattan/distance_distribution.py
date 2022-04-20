import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

# Create a world of cuboids or planar buildings
# There are 35 cuboids (11 in top, 7 right, 11 bottom, 6 left) in the outer ring
# There are 12 cuboids in the inner ring (4 top, 2 right, 3 bottom, 3 left)

# # class Cuboid:
# #     center = np.zeros((3, 1)) # center is at ground level
# #     width = 5
# #     height = 10
# #     breadth = 15

# #     def __init__(self, center, width, height, breadth):
# #         self.center = center
# #         self.width = width
# #         self.height = height
# #         self.breadth = breadth

# #     def shortest_distance_to_point(self, point):
# #         return 0

# #     def o3d_object(self):
# #         o3d_obj = o3d.geometry.PointCloud()

# #         return o3d_obj

# # cuboid1 = Cuboid(np.array([5, 4, 3]), 3, 4, 5)
# # print("Cuboid properties are " + str(cuboid1))


# # Write the distance function

# # Compute the nearest distance to obstacle at various points

# # Add noise to the cuboid parameters

# # Collect the data again

# # Plot the error obtained

# # Remember this in Open3d, 
# # For coordinate frame: red arrow  = x-axis, green arrow = y-axis, blue arrow = z-axis
# # For box (defaults, without any transform, left bottom point lies at (0,0,0)): 
# #   width = along x-axis
# #   height = along y-axis
# #   depth = along z-axis

# def get_box_sdf(center, dims, wp):
#     p = wp - center # transform point to box coordinate frame
#     d = dims/2 # Only along half dims

#     q = abs(p) - d

#     sdf = np.linalg.norm(np.clip(q, 0, None)) + min(q.max(), 0)

#     return sdf

# geometries = []

# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

# box1 = o3d.geometry.TriangleMesh.create_box(width=2.0, height=0.1, depth=1.0)
# box1.translate(np.array([2, 0, 3]).reshape((3, 1)))

# box2 = o3d.geometry.TriangleMesh.create_box(width=3.0, height=0.5, depth=2.0)
# box2.translate(np.array([5, 0, 3]).reshape((3, 1)))

# geometries.append(coord_frame)
# geometries.append(box1)
# # geometries.appen(box2)

# pcd = o3d.geometry.PointCloud()
# pts = np.array([
#     [2, 0, 3],
#     [4, 0, 3],
#     [2, 5, 3],
#     [4, 5, 3]
# ])
# pcd.points = o3d.utility.Vector3dVector(pts)
# geometries.append(pcd)


# print("center of box1 is " + str(box1.get_center()))
# print("center of box2 is " + str(box2.get_center()))


# # Create a 3d pattern of points
# vrange = np.arange(10)
# vx, vy, vz = np.meshgrid(vrange, vrange, vrange)

# vx_pcd = o3d.geometry.PointCloud()
# vx_pts = np.vstack((vx.flatten(), vy.flatten(), vz.flatten())).T
# print("Number of points are: " + str(vx_pts.shape))
# vx_pcd.points = o3d.utility.Vector3dVector(vx_pts)
# geometries.append(vx_pcd)

# o3d.visualization.draw_geometries(geometries)

# # Now call get sdf method for all points and boxes
# box1_gt_sdfs = np.array([get_box_sdf(box1.get_center(), np.array([[2, 0.1, 1]]).T, pt) for pt in vx_pts])
# box1_noisy_sdfs = np.array([get_box_sdf(box1.get_center(), np.array([[2.1, 0.2, 1.1]]).T, pt) for pt in vx_pts])
# print("sdfs shape is " + str(box1_gt_sdfs.shape))

# box2_gt_sdfs = np.array([get_box_sdf(box2.get_center(), np.array([[3, 0.5, 2]]).T, pt) for pt in vx_pts])
# box2_noisy_sdfs = np.array([get_box_sdf(box2.get_center(), np.array([[3.1, 0.6, 2.1]]).T, pt) for pt in vx_pts])

# sdfs_gt = np.minimum(box1_gt_sdfs, box2_gt_sdfs)
# sdfs_noisy = np.minimum(box1_noisy_sdfs, box2_noisy_sdfs)

# sdfs_error = np.abs(sdfs_gt - sdfs_noisy)

# # Now plot the distance distro
# hist, bins = np.histogram(sdfs_error, bins=200, density=True)
# width = 0.7 * (bins[1] - bins[0])
# center = (bins[:-1] + bins[1:]) / 2
# plt.bar(center, hist, align='center', width=width)
# plt.xlabel('error of (d) in meters')
# plt.ylabel('probability of error')
# plt.title('Probability distribution of sdf error w.r.t ground truth sdf')
# plt.show()

















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
    vox_centers = None
    gt_sdfs = None
    geometries = None

    def __init__(self, bound):
        vrange = np.arange(0, bound, 0.4)
        vx, vy, vz = np.meshgrid(vrange, vrange, np.arange(1, bound, 4))
        self.vox_centers = np.vstack((vx.flatten(), vy.flatten(), vz.flatten())).T

    def show(self):
        if self.geometries is None:
            self.geometries = []

            # Add a coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            self.geometries.append(coord_frame)

            # Add vox points
            vox_pcd = o3d.geometry.PointCloud()
            vox_pcd.points = o3d.utility.Vector3dVector(self.vox_centers)
            self.geometries.append(vox_pcd)

            # Add boxes
            self.geometries += [box.mesh for box in self.boxes]

        o3d.visualization.draw_geometries(self.geometries)

    def create_boxes(self, dims, translations):
        # Construct all boxes
        for dim, translation in zip(dims, translations):
            box = Box(dim.flatten())
            box.mesh.translate(translation)

            self.boxes.append(box)

    def compute_gt_sdfs(self):
        for box in self.boxes:
            sdfs = [box.get_sdf(pt) for pt in self.vox_centers]
            
            if self.gt_sdfs is None:
                self.gt_sdfs = sdfs
            
            self.gt_sdfs = np.minimum(sdfs, self.gt_sdfs)
    
    def get_error_sdfs(self, mean=0.1, variance=0.5):
        noisy_sdfs = None

        for box in self.boxes:
            # center_noise = variance * np.random.randn(3, 1) + mean
            # dims_noise = variance * np.random.randn(3, 1) + mean
            center_noise = variance * np.random.random((3, 1)) + mean
            dims_noise = variance * np.random.random((3, 1)) + mean
            ## Uniform distro
            # b = mean + variance
            # a = mean - variance
            # center_noise = (b - a) * np.random.random_sample((3, 1)) + a
            # dims_noise = (b - a) * np.random.random_sample((3, 1)) + a

            sdfs = [box.get_sdf(pt, center_noise, dims_noise) for pt in self.vox_centers]
            
            if noisy_sdfs is None:
                noisy_sdfs = sdfs
            
            noisy_sdfs = np.minimum(sdfs, noisy_sdfs)

        if self.gt_sdfs is None:
            self.compute_gt_sdfs()

        return np.abs(self.gt_sdfs - noisy_sdfs)
    
    def plot_sdf_error_distro(self):
        means = [2,1,0.5,0]
        covariances = [1, 0.6, 0.2]
        errors = np.array([])

        fig, ax = plt.subplots(len(covariances), 1)
        for var_id, variance in enumerate(covariances):
            for mean_id, mean in enumerate(means):
                for i in range(50):
                    errors = np.concatenate((errors, self.get_error_sdfs(mean=mean, variance=variance)))
            
                hist, bins = np.histogram(errors[errors<5], bins=50, density=True)
                width = (1/len(means)) * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2
                ax[var_id].bar(center + (mean_id/len(means)) * 0.5, hist, align='center', width=width, alpha=0.4, label="variance = " + str(variance) + ", mean (m) ="+str(mean))

                ax[var_id].legend(loc='upper right')
                np.savetxt("histogram_data_non_gauss_"+str(var_id)+"_"+str(mean_id)+".txt", errors, delimiter=',')

        plt.title('Probability distribution of sdf error w.r.t ground truth sdf')
        plt.show()

# Create a world
bworld = BoxWorld(10)

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
bworld.create_boxes(np.array(dims), np.array(translations))

# Show the world
bworld.show()

# Compute gt sdfs
bworld.compute_gt_sdfs()

# # Plot all the distros
bworld.plot_sdf_error_distro()
