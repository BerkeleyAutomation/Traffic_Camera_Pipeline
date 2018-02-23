import numpy as np
from scipy.stats import multivariate_normal
import copy
import IPython

from tcp.utils.utils import compute_angle,measure_probability
class Trajectory():

    def __init__(self, initial_pose,config):
        '''
        Initialize trajectory 

        Parameters
        -----------
        initial_pose: dict
        data structure for pose class

        config: Config
        configuration class for traffic intersection 


        '''

        self.initial_time_step = initial_pose['timestep']
        self.class_label = initial_pose['class_label']
        self.list_of_states = [initial_pose]
        self.past_angle = self.compute_original_angle()
        self.config = config

        self.still_on = True

        self.cov = np.array([[ 6.0,  0.0, 0.0],
                           [ 0.0,  6.0, 0.0],
                           [0.0,  0.0,  1.91510572]])


    def get_next_state(self):
        '''
        Return the pose for the last state of the trajectory
        
        Return
        ---------
        np.array, size 2 for (x,y) pose 
        bool, True if the list isn't empty

        '''

        if len(self.list_of_states) == 0:
            return None, False

        state = self.list_of_states.pop(0)

        return state['pose'],True


    def compute_original_angle(self):
        '''
        Checks the lane of the first state to return the angle that corresponds it to 
        driving straight along the direction of the lane

        Return
        ---------
        float, current angle range [-pi,pi]

        '''

        lane = self.list_of_states[0]['lane']
        if lane is None:
            return 0.0

        if lane['lane_index'] == 0:
            return np.pi/2

        elif lane['lane_index'] == 1:
            return -np.pi/2

        elif lane['lane_index'] == 2:
            return np.pi

        elif lane['lane_index'] == 3:
            return 0.0


    def return_last_state_pos(self):
        '''
        Returns last state's pose

        Returns
        ---------
        np.array, size 2 for (x,y) pose 

        '''

        state = self.list_of_states[-1]
        return state['pose']


    def append_to_trajectory(self,datum):
        '''
        Append new pose to trajectory and updates the past angle to have
        the angle computed in compute_probability
        '''

        self.list_of_states.append(datum)
        self.past_angle = self.curr_angle


    def not_valid(self,current_timestop):
        '''
        Checks if the current trajectory has been updated in a while, 
        if it hasn't marks the trajectory as complete. 

        Parameters
        ----------
        current_timestep: int
          the current timestep of the traffic camera

        Returns
        ------------
        bool, True if the trajectory is valid and False if not. 
        '''

        last_state = self.list_of_states[-1]

        if current_timestop - last_state['timestep'] > self.config.time_limit:
            return True

        return False
          

    def compute_new_angle(self):
        '''
        Compute the current angle of the trajectories, with respect 
        to the evaluated state. 

        Returns
        ------------
        float, current angle in range [-pi,pi] 
        '''

        pos = self.return_last_state_pos()

        angle = compute_angle(self.curr_state,pos)

        return angle


    def compute_probability(self,state):
        '''
        Compute the log probability of the proposed state corresponding to this
        trajecotry 

        Returns
        ------------
        float, range [-inf,0] where 0 corrresponds to more probable 
        '''

        pos = self.return_last_state_pos()
        self.curr_state = [state['pose'][0],state['pose'][1]]

        curr_angle = self.compute_new_angle()
        
        mean = np.array([pos[0],pos[1],self.past_angle])        
        state_full = np.array([self.curr_state[0],self.curr_state[1],curr_angle])

        var = measure_probability(self.cov,mean,state_full)
       
        self.curr_angle = curr_angle
        
        return var
