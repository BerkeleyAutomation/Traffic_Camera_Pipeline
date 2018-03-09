import sys, os
import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import pickle
import skimage.transform
import cv2
import IPython
import pygame
from random import shuffle

# import the necessary packages
import argparse
import cv2

class CameraLabeler():

    def __init__(self):
        ''' 
        Initialize the VizRegristation Class

        Parameters
        ----------
        cnfig: Config
        configuration class for traffic intersection

        '''
        self.point_found = False


 
    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
        global refPt, cropping
     
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point = np.array([x, y])
            self.point_found= True


    def initalize_mouse_handler(self):

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", self.click_and_crop)


    def plot_correspondence(self,pose, image):

        py = int(pose[1])
        px = int(pose[0])
        image[py-5:py+5,px-5:px+5,:] = 255

        return image

    def cam_labeler(self,image,correspondence = None):
        '''
        Visualize the sperated trajecotries in the simulator and can also visualize the matching images

        Parameter
        ------------
        trajectories: list of Trajectory 
        A list of Trajectory Class

        plot_traffic_images: bool
        True if the images from the traffic cam should be shown alongside the simulator 
        '''
        self.initalize_mouse_handler()

        
        image = self.plot_correspondence(correspondence,image)
       
        
        
        while True:
            cv2.imshow("image", image)
            cv2.waitKey(10)
      

            if self.point_found:
               self.point_found = False

    

               cv2.destroyAllWindows()


               return self.point

        

 
