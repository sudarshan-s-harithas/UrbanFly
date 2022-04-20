"""Utilies to create a box world"""
import numpy as np

def normalize(np_vec):
    return np_vec / np.linalg.norm(np_vec)

def cross(np_vec1, np_vec2):
    return np.cross(np_vec1.flatten(), np_vec2.flatten()).reshape((3, 1))

def homogenous(np_arr):
    a = np_arr.flatten()

    return np.array([a[0], a[1], a[2], 1.0]).reshape((4, 1))

class Box:
    def __init__(self, vertices):
        self.vertices = vertices
        self.front_plane = np.zeros((4, 1))
        self.right_plane = np.zeros((4, 1))
        self.back_plane = np.zeros((4, 1))
        self.left_plane = np.zeros((4, 1))
        self.top_plane = np.zeros((4, 1))
        self.bottom_plane = np.zeros((4, 1))

        self.init_from_vertices()

    def init_from_vertices(self):
        edge20 = self.vertices[0] - self.vertices[2]
        edge26 = self.vertices[6] - self.vertices[2]
        edge23 = self.vertices[3] - self.vertices[2]

        self.front_plane[:3] = normalize(cross(edge20, edge26))
        self.top_plane[:3] = normalize(cross(edge23, edge20))
        self.right_plane[:3] = normalize(cross(edge26, edge23))
        self.back_plane[:3] = -self.front_plane[:3]
        self.bottom_plane[:3] = -self.top_plane[:3]
        self.left_plane[:3] = -self.right_plane[:3]

        self.front_plane[3, 0] = -self.front_plane[:3].T.dot(self.vertices[2])
        self.top_plane[3, 0] = -self.top_plane[:3].T.dot(self.vertices[2])
        self.right_plane[3, 0] = -self.right_plane[:3].T.dot(self.vertices[2])
        self.back_plane[3, 0] = -self.back_plane[:3].T.dot(self.vertices[3])
        self.bottom_plane[3, 0] = -self.bottom_plane[:3].T.dot(self.vertices[6])
        self.left_plane[3, 0] = -self.left_plane[:3].T.dot(self.vertices[0])

    def get_sdf(self, point):
        point_ = homogenous(point).flatten()

        distance = max([
            self.front_plane.T.dot(point_),
            self.top_plane.T.dot(point_),
            self.right_plane.T.dot(point_),
            self.back_plane.T.dot(point_),
            self.bottom_plane.T.dot(point_),
            self.left_plane.T.dot(point_)
        ])
        return distance
    
    def get_face_planes(self):
        face_planes = np.hstack((
            self.front_plane,
            self.right_plane,
            self.back_plane,
            self.left_plane,
            self.top_plane,
            self.bottom_plane
        ))
        
        return face_planes.T 

class BoxWorld:
    boxes = []
    gt_sdfs = None
    geometries = None

    def __init__(self, vertices_msg):
        self.boxes = []
        vertices = self.points_to_numpy(vertices_msg.points)
        self.init_boxes_from_vertices(vertices)

    def points_to_numpy(self, points):
        point_list = []
        for point in points:
            point_list.append([point.x, point.y, point.z])

        return np.array(point_list)

    def init_boxes_from_vertices(self, vertices):
        for vertex_idx in range(0, vertices.shape[0], 8):
            self.boxes.append(Box(vertices[vertex_idx:vertex_idx+8, :]))

    def is_point_inside(self, point):
        min_sdf = 10000
        for box in self.boxes:
            min_sdf = min(box.get_sdf(point), min_sdf)

        if min_sdf <= 1:
            return True
        else:
            return False

    def is_colliding_trajectory(self, x_s, y_s, z_s):
        pts = np.vstack((x_s.flatten(), y_s.flatten(), z_s.flatten())).T
        
        for pt in pts:
            if self.is_point_inside(pt):
                return True

        return False

    def get_point_cost(self, point):
        min_sdf = 10000
        for box in self.boxes:
            min_sdf = min(box.get_sdf(point), min_sdf)

        return min_sdf
    
    def get_trajectory_cost(self, x_s, y_s, z_s):
        pts = np.vstack((x_s.flatten(), y_s.flatten(), z_s.flatten())).T
        trajectory_cost = 0.0
        is_colliding = False
        
        for pt in pts:
           point_cost = self.get_point_cost(pt)
           trajectory_cost += point_cost

           if (point_cost <= 1.0) and not is_colliding:
               is_colliding = True

        return trajectory_cost, is_colliding 
    
    def get_plane_params(self):
        params = np.zeros((0, 4))
        
        for box in self.boxes:
            params = np.vstack((params, box.get_face_planes()))
            
        return params
