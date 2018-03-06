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

from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent#, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight

import colorlover as cl



class PygameLabeler():

    def __init__(self,config):

        self.config = config
    
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




    def sim_labeler(self):
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
       
        

        
        while True:
            values = pygame.mouse.get_pressed()

            self.env._render(traffic_trajectories = self.config.registration_points)

            if values[0]:
                pose = pygame.mouse.get_pos()

                return np.array([pose[0],pose[1]])/0.8



 
