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


    def load_frames(self, video_name):
        '''
        Load label images to be plotted 
        '''
        if video_name is None:
            print 'Visualizer failed to load frames. Video name not provided.'
            return

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


    def get_color_template(self):
        ''''
        Returns a spectrum of colors that intrepret between two different spectrums 
        The goal is to have unique color for each trajectories
        '''

        colors = [tuple(map(lambda c: int(255 * c), color)) for color in sns.color_palette("Set1", 8)]
        return colors

    def visualize_trajectory_dots(self, trajectories, filter_class=None, plot_traffic_images=False, video_name=None):
        '''
        Visualize the sperated trajecotries in the simulator and can also visualize the matching images

        Parameter
        ------------
        trajectories: list of Trajectory 
        A list of Trajectory Class

        plot_traffic_images: bool
        True if the images from the traffic cam should be shown alongside the simulator 
        '''
        def get_way_points(class_label):
            active_trajectories = []
            way_points = []

            color_template = self.get_color_template()
            color_index = 0
            for t in range(self.config.vz_time_horizon):
                for traj_index, traj in enumerate(trajectories):
                    if traj.class_label != class_label:
                        continue

                    if t == traj.initial_time_step:
                        color_match = {'trajectory': traj, 
                                       'color_template': color_template[color_index]}
                        active_trajectories.append(color_match)
                        color_index += 1
                        if color_index == len(color_template):
                            color_index %= len(color_template)
                            shuffle(color_template)
                
                for traj_index, traj in enumerate(active_trajectories):
                    traj = traj['trajectory']
                    if traj.class_label != class_label:
                        continue
                    poses, valid = traj.get_states_at_timestep(t)

                    if valid:
                        for pose in poses:
                            w_p = [pose, active_trajectories[traj_index]['color_template']]
                            way_points.append(w_p)
            return way_points


        self.initalize_simulator()
        if plot_traffic_images:
            self.load_frames(video_name)
    
        ###Render Images on Simulator and Traffic Camera
        if filter_class is None:
            car_way_points = get_way_points('car')
            pedestrian_way_points = get_way_points('pedestrian')
            way_points = car_way_points + pedestrian_way_points
            self.env._render(traffic_trajectories=way_points)
        else:
            assert filter_class == 'car' or filter_class == 'pedestrian', 'Invalid filter_class: should be car or pedestrian.'
            way_points = get_way_points(filter_class)
            self.env._render(traffic_trajectories=way_points)
        
        if plot_traffic_images:
            cv2.imshow('img',self.imgs[t])
            cv2.waitKey(30)
            t+=1

        return

    def visualize_homography_points(self):
        ''' 
        Plot the correspnding homography ponts
        Assumes load_frames has been called
        '''

        self.initalize_simulator()
        img = self.imgs[0]

        for i in range(3):

            point = self.config.street_corners[i,:]

            img[point[1]-5:point[1]+5,point[0]-5:point[0]+5,:]=255

        waypoints = []

        for i in range(3):
            point = self.config.simulator_corners[i,:]
            waypoints.append(point)

        while True:
            self.env._render(waypoints = waypoints)     
            cv2.imshow('img',img)
            cv2.waitKey(30)
