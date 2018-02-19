import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import pickle
import skimage.transform
import cv2

from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight
import IPython



class Homography():

    def __init__(self,config,vz_debug = None):

        self.config = config
        # Four corners of the interection, hard-coded in camera space
        self.corners = self.config.street_corners
        # Four corners of the intersection, hard-coded in transformed space
        self.st_corners = self.config.simulator_corners

        self.tf_mat = skimage.transform.ProjectiveTransform()
        self.tf_mat.estimate(self.st_corners, self.corners)

        self.vz_debug = vz_debug

    def determine_lane(self,point):

        for i in range(len(self.config.lanes)): 

            lane = self.config.lanes[i]

            if lane.contains_point(point):
                side = lane.side_of_road(point)
                return {'lane_index':i, 'road_side':side}

            
        return None

    def transform_trajectory(self,trajectory):
        tf_traj = []
        for frame in trajectory:
            new_frame = []
            for obj in frame:
                x,y,cls_label,t = obj

                x, y = self.config.alberta_img_dim[0] * x, self.config.alberta_img_dim[1]* y

                tx, ty = self.tf_mat.inverse(np.array((x, y)))[0]
                # tx = tx * self.config.sim_scale[0]
                # ty = ty * self.config.sim_scale[1]

                pose = np.array([tx,ty])

                new_obj = {'pose':pose,'class_label':cls_label,
                    'timestep':t,
                    'lane':self.determine_lane(pose)}
                new_frame.append(new_obj)

            tf_traj.append(new_frame)

        return tf_traj

    def test_homography(self):
      
        for i in range(4):

            point = self.config.street_corners[i,:]
            x,y = point
               
            tx, ty = self.tf_mat.inverse(np.array((x, y)))[0]

           

    def test_camera_point(self,trajectory):
      
        for frame in trajectory:
            for obj in frame:
                x,y,cls_label,t = obj

                x, y = int(self.config.alberta_img_dim[0] * x), int(self.config.alberta_img_dim[1]* y)
                self.vz_debug.visualize_camera_point(x,y,t)
               
            

