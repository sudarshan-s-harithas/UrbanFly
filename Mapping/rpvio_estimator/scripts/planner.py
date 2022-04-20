#!/usr/bin/env python
from numpy.lib.function_base import append
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform.rotation import Rotation

import cvxpy as cp

import numpy as np
from box_world import BoxWorld

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

is_init_path = False

class Planner:
    num_of_paths = 50 #number of paths
    num_of_way_points = 30 #number of waypoints in each waypoint
    tic = np.zeros(3)
    ric = np.eye(3)
    ti = np.zeros(3)
    ri = np.eye(3)
    global_goal = np.array([23.0, -5.0, 5.0])

    def __init__(self, vertices_msg, odometry_msg, local_goal_pub, local_stomp_pub, feasible_path_pub, free_cloud_pub, colliding_cloud_pub):
        self.vertices_msg = vertices_msg
        self.odometry_msg = odometry_msg
        
        self.read_cam_imu_transform()
        self.parse_odometry_msg(odometry_msg)

        self.map = BoxWorld(vertices_msg)

        self.local_goal = self.world2cam(self.global_goal)
        self.local_goal[1] = 0.0
        #print("Local goal is: ")
        #print(self.local_goal)

        self.local_goal_pub = local_goal_pub
        self.local_stomp_pub = local_stomp_pub
        self.feasible_path_pub = feasible_path_pub
        self.free_cloud_pub = free_cloud_pub
        self.colliding_cloud_pub = colliding_cloud_pub
    
    def read_cam_imu_transform(self):
        fs = cv2.FileStorage("../../config/rpvio_sim_config.yaml", cv2.FILE_STORAGE_READ)
        self.ric = np.array(fs.getNode("extrinsicRotation").mat()).reshape((3, 3))
        self.tic =  np.array(fs.getNode("extrinsicTranslation").mat()).reshape((3, 1))
        
        #print("Read cam imu transform")
        #print("tic: ")
        #print(self.tic)
        #print("ric: ")
        #print(self.ric)

    def parse_odometry_msg(self, odometry_msg):
        trans = odometry_msg.pose.pose.position
        rot = odometry_msg.pose.pose.orientation
        self.ti = np.array([trans.x, trans.y, trans.z]).reshape((3, 1))
        self.ri = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()

    def world2cam(self, world_vector):
        world_vector = world_vector.reshape((3, 1))
        local_vector = self.ri.T.dot(world_vector) - self.ri.T.dot(self.ti)
        cam_vector = self.ric.T.dot(local_vector) - self.ric.T.dot(self.tic)
        return cam_vector.flatten()

    def cam2world(self, local_vector):
        local_vector = local_vector.reshape((3, 1))
        imu_vector = self.ric.dot(local_vector) + self.tic
        world_vector = self.ri.dot(imu_vector) + self.ti
        return world_vector.flatten()

    def compute_paths(self):
        self.compute_stomp_paths()

    def compute_stomp_paths(self):
        num_goal = self.num_of_paths
        num = max(int(np.linalg.norm(self.local_goal)), 3)
        self.num = num

        x_init =  0.0
        y_init =  0.0
        z_init =  0.0

        x_des_traj_init = x_init
        y_des_traj_init = y_init
        z_des_traj_init = z_init

        ############################################################## Hyperparameters
        t_fin = 5

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

        x_fin = self.local_goal[0] 
        y_fin = self.local_goal[1]
        z_fin = self.local_goal[2] 

        t_interp = np.linspace(0, t_fin, num)
        x_interp = x_des_traj_init + ((x_fin-x_des_traj_init)/t_fin) * t_interp
        y_interp = y_des_traj_init + ((y_fin-y_des_traj_init)/t_fin) * t_interp
        z_interp = z_des_traj_init + ((z_fin-z_des_traj_init)/t_fin) * t_interp

        # A_mat = A
        # print(temp_1.shape)
        # print(A_mat.shape)
        R = np.dot(A_mat.T, A_mat)
        mu = np.zeros(num)
        cov = np.linalg.pinv(R)

        # print(R.shape)
        ################# Gaussian Trajectory Sampling
        eps_kx = np.random.multivariate_normal(mu, 0.06*cov, (num_goal, ))
        eps_ky = np.random.multivariate_normal(mu, 0.06*cov, (num_goal, ))
        eps_kz = np.random.multivariate_normal(mu, 0.06*cov, (num_goal, ))

        self.x_samples = x_interp+eps_kx
        self.y_samples = y_interp+0.0*eps_ky
        self.z_samples = z_interp+eps_kz

    def publish_paths(self):
        self.publish_local_goal()
        self.publish_local_stomp_paths()

    def publish_local_goal(self):
        goal_pc = PointCloud()
        goal_pc.header = self.vertices_msg.header
        goal_pc.points.append(self.to_ros_point(self.global_goal))
        self.local_goal_pub.publish(goal_pc)
    
    def publish_local_stomp_paths(self):
        ma = MarkerArray()
        
        optimal_line_strip = None
        is_optimal_colliding = True
        max_sdf_cost = -100000000

        good_trajs = []

        # Add a line set marker for each path
        for p in range(self.x_samples.shape[0]):
            line_strip = Marker()
            line_strip.header = self.vertices_msg.header
            
            line_strip.pose.orientation.w = 1.0

            line_strip.id = p+2
            line_strip.type = Marker.LINE_STRIP

            line_strip.scale.x = 0.03

            line_strip.color.r = 1.0
            line_strip.color.a = 0.7

            is_colliding = False
            traj_cost = 0.0

            pts = []
            for w in range(self.x_samples.shape[1]):
                pt = np.array([
                    self.x_samples[p, w],
                    self.y_samples[p, w],
                    self.z_samples[p, w]
                ]).flatten()
                w_pt = self.cam2world(pt).flatten() 

                pts += [[pt[0], pt[1], pt[2]]]
                
                line_strip.points.append(self.to_ros_point(w_pt))
                point_cost = self.map.get_point_cost(pt)
                if point_cost < 1.0:
                    is_colliding = True

                traj_cost += point_cost

            # if not is_colliding:
                # good_trajs += pts

            if (traj_cost > max_sdf_cost) and not is_colliding:
                max_sdf_cost = traj_cost
                is_optimal_colliding = False
                optimal_line_strip = line_strip
                good_trajs = pts

            ma.markers.append(line_strip)

        if is_optimal_colliding:
            optimal_line_strip = None

        # consider all collision-free trajectories
        num = self.num
        traj_num = int(len(good_trajs)/num)

        #print("Number of collision-free trajectories are : ", str(traj_num))

        planes = self.map.get_plane_params()
        num_of_planes = planes.shape[0]
        #print("Number of planes are : ", str(num_of_planes))

        optimized_traj = None

        if traj_num > 0 and num_of_planes > 0 and not is_optimal_colliding:

            total_time = 0.0
            for i in range(int(len(good_trajs)/num)):
                k = i * num
                trajectory1 = np.array(good_trajs)[k:k+num, :3]
                
                normals = planes[:, :3]
                ds = planes[:, -1:] @ np.ones((1, trajectory1.shape[0]))

                # firstly compute the plane collision matrix
                collision_mat = normals @trajectory1.T + ds
                #print("Collision matrix size is ", collision_mat.shape)

                bin_collision_mat = np.zeros(collision_mat.shape)
                bin_collision_mat[collision_mat > 0] = 1.0

                x = cp.Variable((num, 3))
                x.value = trajectory1
                constraint = [x[0, :] == trajectory1[0, :]]
                constraint += [x[num-1, :] == trajectory1[num-1, :]]
                constraint += [cp.multiply(bin_collision_mat, normals@x.T + ds) >= 0.0]

                # # plane = cp.Parameter((3, 1))
                # # plane.value = r_plane.reshape((3, 1))

                # # onz = cp.Parameter((num, 1))
                # # onz.value = np.ones((num, 1))
                # # # constraint += [x @ r_plane - 20*onz >= 0]
                dmat = np.diff(np.diff(np.eye(num)))
                print("initial cost: ", np.linalg.norm(dmat.T@x.value)) 
                obj = cp.Minimize(cp.sum_squares(dmat.T@x))
                # obj = cp.Minimize(cp.sum_squares(cp.multiply(bin_collision_mat, normals@x.T + ds)))
                problem = cp.Problem(obj, constraint)
                try:
                    problem.solve()
                    #print('Solving ...: ', problem.solve())
                    #print('Status of the problem: ', problem.status)
                    total_time += problem.solver_stats.solve_time
                    # print('x value is ', x.value)

                    # pcd2.colors = o3d.utility.Vector3dVector(np.array(traj_colors))

                    if problem.status == 'optimal':
                        optimized_traj = x.value
                        print("optimal cost: ", np.linalg.norm(dmat.T@x.value)) 
                        print('Time taken to solve: ', problem.solver_stats.solve_time)
                except Exception as e:
                    print(e)

            #print('Total time taken for individual optimization', total_time)

        if optimized_traj is not None:
            line_strip = Marker()
            line_strip.header = self.vertices_msg.header
            
            line_strip.pose.orientation.w = 1.0

            line_strip.id = 9998
            line_strip.type = Marker.LINE_STRIP

            line_strip.scale.x = 0.06

            line_strip.color.g = 1.0
            line_strip.color.a = 0.7

            for way_pt in optimized_traj:
                line_strip.points.append(self.to_ros_point(self.cam2world(way_pt)))
        
            optimal_line_strip = line_strip

        elif not is_optimal_colliding and optimal_line_strip is not None:
            optimal_line_strip.color.r = 0.0
            optimal_line_strip.color.g = 0.0
            optimal_line_strip.color.b = 1.0
            optimal_line_strip.scale.x = 0.05
            optimal_line_strip.color.a = 1.0

        if optimal_line_strip is not None:
            optimal_line_strip.id = 9999
            ma.markers.append(optimal_line_strip)

            global is_init_path
            if (not is_init_path) or (optimized_traj is not None):
                if optimized_traj is not None:
                    is_init_path = True
                feasible_cloud = PointCloud()
                feasible_cloud.header = self.vertices_msg.header

                for point in optimal_line_strip.points:
                    #pt = self.from_ros_point(point)
                    #w_pt = self.cam2world(pt)
                    #w_point = self.to_ros_point(w_pt.flatten())

                    feasible_cloud.points.append(point)

                self.feasible_path_pub.publish(feasible_cloud)

        self.local_stomp_pub.publish(ma)

        colliding_points = PointCloud()
        colliding_points.header = self.odometry_msg.header
        free_points = PointCloud()
        free_points.header = self.odometry_msg.header

        for i in range(-10, 10, 1):
            for j in range(-10, 10, 1):
                c_pt = np.array([i, 0.0, j])
                is_colliding = False

                point_cost = self.map.get_point_cost(c_pt)
                if point_cost < 1.0:
                    is_colliding = True
                
                if is_colliding:
                    colliding_points.points.append(self.to_ros_point(self.cam2world(c_pt)))
                else:
                    free_points.points.append(self.to_ros_point(self.cam2world(c_pt)))
        
        self.free_cloud_pub.publish(free_points)
        self.colliding_cloud_pub.publish(colliding_points)

    def from_ros_point(self, ros_point):
        return np.array([ros_point.x, ros_point.y, ros_point.z])

    def to_ros_point(self, point):
        ros_point = Point()
        ros_point.x = point[0]
        ros_point.y = point[1]
        ros_point.z = point[2]

        return ros_point
