import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import pickle
import skimage.transform
import cv2
import copy
import IPython
import numpy.linalg as LA
from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight
from tcp.registration.trajectory import Trajectory


from tcp.utils.utils import compute_angle,euclidean_angle_distance
class CovEst():

    def __init__(self,config):

        self.config = config
    

    def compute_angles(self,frame,prev_frame):
     
          pos = frame[0]['pose']
          pos_b = prev_frame[0]['pose']

          return compute_angle(pos,pos_b)

    def compute_variance(self,frame,prev_frame,prev_angle):

        x_delta = frame[0]['pose'][0] - prev_frame[0]['pose'][0]
        y_delta = frame[0]['pose'][1] - prev_frame[0]['pose'][1]
        angle_delta = euclidean_angle_distance(self.compute_angles(frame,prev_frame), prev_angle)

        return np.array([[x_delta,y_delta,angle_delta]])
        

    def compute_covariance(self,trajectories):

        covaraince_matrix = np.zeros([3,3])

        N = float(len(trajectories))


        prev_frame = trajectories[0]
        prev_angle = np.pi/2
        for frame in trajectories[1:]:

            variance = self.compute_variance(frame,prev_frame,prev_angle)
        
            covaraince_matrix += np.dot(variance.T,variance)

            prev_angle = self.compute_angles(frame,prev_frame)
            prev_frame = frame

            print prev_angle


           
        
        return covaraince_matrix/N










        


