import sys, os
import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import pickle
import pygame
import skimage.transform
import cv2
import IPython
from random import shuffle

from tcp.registration.camera_labeler import CameraLabeler

from sklearn.linear_model import LinearRegression

class AddOffset():

    def __init__(self):
        ''' 
        Initialize the VizRegristation Class

        Parameters
        ----------
        cnfig: Config
        configuration class for traffic intersection

        '''


        self.data = pickle.load(open('homography_data/offset_data.p','r'))

        self.model = LinearRegression()

        X = self.data['cam_points']

        Y = self.data['sim_points']
       
        self.model.fit(X,Y)



    def add_offset(self,pose):
        pose = np.array([pose])

        return self.model.predict(pose)[0,:]
        



 
