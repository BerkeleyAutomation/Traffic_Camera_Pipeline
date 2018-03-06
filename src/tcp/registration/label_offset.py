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

from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent#, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight

from tcp.registration.camera_labeler import CameraLabeler



import colorlover as cl



class LabelOffset():

    def __init__(self,cnfg):
        ''' 
        Initialize the VizRegristation Class

        Parameters
        ----------
        cnfig: Config
        configuration class for traffic intersection

        '''

        self.config = cnfg

        self.cl = CameraLabeler()


    def load_frames(self,video_name):
        '''
        Load label images to be plotted 
        '''

       
        self.imgs = []
        for i in range(self.config.hm_training_data):
            debug_img_path = os.path.join(self.config.homography_training_data, video_name+'_images')
            img = cv2.imread(os.path.join(debug_img_path, '%s_%07d.jpg' % (video_name, i)))
            self.imgs.append(img)



    def record_label(self,obj,pose):
        
        cam_pose = obj['cam_pose']

        self.camera_poses.append(cam_pose)

        self.simulator_poses.append(pose)

        print "got here"




    def data_point_check(self,t):
        ''''
        Returns a spectrum of colors that intrepret between two different spectrums 
        The goal is to have unique color for each trajectories

        Parameter
        ------------
        num_trajectories: int
        Number of Trajectories to plot 
        '''

        valid_data = [25,458]

        if t in valid_data:
            return True
        else:
            return False




    def label_trajectories(self, trajectories, video_name=None):
        '''
        Visualize the sperated trajecotries in the simulator and can also visualize the matching images

        Parameter
        ------------
        trajectories: list of Trajectory 
        A list of Trajectory Class

        plot_traffic_images: bool
        True if the images from the traffic cam should be shown alongside the simulator 
        '''
        
        self.load_frames(video_name)
        

        active_trajectories = []
        self.camera_poses = []
        self.simulator_poses = []
        
    

        
        for traj in trajectories:

            if traj.initial_time_step < self.config.hm_training_data:
               
                for obj in traj.list_of_states:
                   

                    if self.data_point_check(obj['timestep']):
                        pose = self.cl.cam_labeler(self.imgs[obj['timestep']],obj['cam_pose'])

                        self.record_label(obj,pose)
                   


        data = {'cam_points':self.camera_poses, 'sim_points':self.simulator_poses}

        pickle.dump(data, open('homography_data/offset_data.p','wb'))
            






 
