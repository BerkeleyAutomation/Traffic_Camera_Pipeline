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





class InitLabeler():

    def __init__(self,config):

        self.config = config
        


    def stop_for_label(self):
        ''''
        if a new car is present the user should press some button 
        and then stop for some label

        Returns:
        ----------
        Bool, True if a label is present false otherwise 

        '''


        return False

    def label_image(self,img,bboxes):

        '''
        Takes an image and the current bbboxes
        return the indices of the bounding box corresponding to the new car

        '''


        


        self.bbox_index = 0

    def get_point(self):

        return self.pixel_label
       









        


