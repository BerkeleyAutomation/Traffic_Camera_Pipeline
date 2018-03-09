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

from tcp.registration.trajectory import Trajectory


class ObsFiltering():

    def __init__(self, config,label = 'car'):
        ''''
        Initializes ObsFiltering class

        Parameters
        -----------
        config: Config
        configuration class for traffic intersection
        '''

        self.config = config
        self.old_trajectories = []
        

    def clear_trajectories(self, current_timestep):
        '''
        Moves the inactive trajectories (i.e. ones that haven't been updated in a while)
        to the old_trajectories list and then removes the trajectories from the current one

        Parameters
        -----------
        current_timestep: int
        corresponds to the current timestep. 

        '''

        elements_to_delete = []

        for i in range(len(self.trajectories)):
            if self.trajectories[i].not_valid(current_timestep):
                elements_to_delete.append(i)

        elements_to_delete.reverse()

        for indx in elements_to_delete:
            traj = self.trajectories[indx]
            self.old_trajectories.append(traj)
            del self.trajectories[indx]

    def select_highest_proposal(self, state):
        '''
        Computes the trajectory that has the highest probability of corresponding to 
        the given new state

        Parameters
        -----------
        state, dict
        data structure for pose class 

        Returns
        ----------
        Trajectory Class (Pass by Reference)

        '''

        highest_val = -np.inf
        best_candidate = None

        for traj in self.trajectories:
            val = traj.compute_probability(state)
            if highest_val < val:
                highest_val = val
                best_candidate = traj
        return best_candidate


    def add_observations_to_trajectories(self, frame):
        ''''
        Add the current frame to the trajectories

        Parameters
        ---------------
        frame: list of dicts
        A list of the current states at the corresponding timesteps

        '''

        for obj in frame:


            if  obj['is_initial_state']:
                new_trajectory = Trajectory(obj,self.config)
                self.trajectories.append(new_trajectory)
            else:
                traj = self.select_highest_proposal(obj)
                # assert traj is not None, 'No existing trajectories to append to.'
                if traj is not None:
                    traj.append_to_trajectory(obj)


    def heuristic_label(self, trajectories):

        ''''
        Recovers the seperate trajectories given the postiional positions from the 
        traffic camera 

        Parameters
        ---------------
        trajectories: a list of tuple frames
        A list of frames, which is a list state dictionaries for each timestep

        '''

        self.trajectories = []
        self.old_trajectories = []
        for frame in trajectories:
            self.add_observations_to_trajectories(frame)
            self.clear_trajectories(frame[0]['timestep'])

        trajectories = self.trajectories + self.old_trajectories

        for traj in trajectories:
            traj.prune_points_near_edge()
            
        return trajectories
