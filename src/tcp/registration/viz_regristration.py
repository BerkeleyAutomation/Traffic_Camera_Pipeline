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
from random import shuffle

from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent#, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight

import seaborn as sns



class VizRegristration():

    def __init__(self,cnfg):
        ''' 
        Initialize the VizRegristation Class

        Parameters
        ----------
        cnfig: Config
        configuration class for traffic intersection

        '''

        self.config = cnfg



    def load_frames(self,video_name):
        '''
        Load label images to be plotted 
        '''

       
        self.imgs = []
        for i in range(self.config.vz_time_horizon):

            debug_img_path = os.path.join(self.config.save_debug_img_path, video_name)
            img = cv2.imread(os.path.join(debug_img_path, '%s_%07d.jpg' % (video_name, i)))
            self.imgs.append(img)

    def initalize_simulator(self):

        '''
        Initializes the simulator
        '''

        self.vis = uds.PyGameVisualizer((800, 800))

        # Create a simple-intersection state, with no agents
        self.init_state = uds.state.SimpleIntersectionState(ncars=0, nped=0, traffic_lights=True)

        # Create the world environment initialized to the starting state
        # Specify the max time the environment will run to 500
        # Randomize the environment when env._reset() is called
        # Specify what types of agents will control cars and traffic lights
        # Use ray for multiagent parallelism
        self.env = uds.UrbanDrivingEnv(init_state=self.init_state,
                                  visualizer=self.vis,
                                  max_time=500,
                                  randomize=False,
                                  agent_mappings={Car:NullAgent,
                                                  TrafficLight:TrafficLightAgent},
                                  use_ray=False
        )

        self.env._reset(new_state=self.init_state)

        IPython.embed()


    def get_color_template(self):
        ''''
        Returns a spectrum of colors that intrepret between two different spectrums 
        The goal is to have unique color for each trajectories
        '''

        colors = [tuple(map(lambda c: int(255 * c), color)) for color in sns.color_palette("Set1", 8)]
        return colors

    def visualize_trajectory_dots(self, trajectories, plot_traffic_images=False, video_name=None,render_surface = None):
        '''
        Visualize the sperated trajecotries in the simulator and can also visualize the matching images

        Parameter
        ------------
        trajectories: list of Trajectory 
        A list of Trajectory Class

        plot_traffic_images: bool
        True if the images from the traffic cam should be shown alongside the simulator 
        '''
        self.initalize_simulator()
        if plot_traffic_images:
            self.load_frames(video_name)
        

        active_trajectories = []
        way_points = []

        color_template = self.get_color_template()
        color_index = 0
        for t in range(self.config.vz_time_horizon):

            for traj_index in range(len(trajectories)):

                traj = trajectories[traj_index]
                if t == traj.initial_time_step:    
                    color_match = [traj, color_template[color_index]]
                    active_trajectories.append(color_match)
                    color_index += 1
                    if color_index == len(color_template):
                        color_index %= len(color_template)
                        shuffle(color_template)
            
            for traj_index in range(len(active_trajectories)):

                traj = active_trajectories[traj_index][0]
                poses, valid = traj.get_states_at_timestep(t)

                if valid:
                    for pose in poses:
                        w_p = [pose, active_trajectories[traj_index][1]]
                        way_points.append(w_p)

            if plot_traffic_images and not render_surface == None:
                surface = render_surface(self.imgs[t])
            else:
                surface = None

            if plot_traffic_images:
                cv2.imshow('img',self.imgs[t])
                cv2.waitKey(10)
                t+=1

            

            ###Render Images on Simulator and Traffic Camera 
            self.env._render(traffic_trajectories = way_points)
            
            


        return

    def visualize_homography_points(self,hm):
        ''' 
        Plot the correspnding homography ponts
        Assumes load_frames has been called
        '''

        self.initalize_simulator()
        img = self.imgs[0]

        for i in range(4):

            point = hm.cc[i]

            img[point[1]-5:point[1]+5,point[0]-5:point[0]+5,:]=255

        img = hm.apply_homography_on_img(img)
        

        waypoints = []

        for i in range(4):
            point = hm.sc[i]
            waypoints.append([point,(0,255,0)])

        while True:
            self.env._render(waypoints = waypoints,transparent_surface = img)     
            
