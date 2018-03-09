import numpy as np
import copy
import IPython

from scipy.stats import multivariate_normal
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

from tcp.utils.utils import compute_angle, measure_probability, is_valid_lane_change

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

        self.cov = np.array([[6.0, 0.0, 0.0, 0.0],
                             [0.0, 6.0, 0.0, 0.0],
                             [0.0, 0.0, 1.57, 0.0],
                             [0.0, 0.0, 0.0, 150000]])

        #SHould be 15e4?

        self.probability_list = []


    def get_valid_states(self):
        """
        Returns a list of states whose poses are within the UDS window.
        """
        def in_uds_range(pose):
            return pose[0] >= 0 and pose[0] <= 1000 and pose[1] >= 0 and pose[1] <= 1000

        return [state_dict for state_dict in self.list_of_states if in_uds_range(state_dict['pose'])]

    def get_initial_state_timesteps(self):
        return [state_dict['timestep'] for state_dict in self.list_of_states if state_dict['is_initial_state']]

    def get_states_at_timestep(self, t):
        return [state_dict for state_dict in self.list_of_states if state_dict['timestep'] == t]

    def get_poses_at_timestep(self, t):
        '''
        Returns a list of poses corresponding to timestep t in the trajectory.
        
        Return
        ---------
        list of np.array, each size 2 for (x,y) pose 
        bool, True if the list isn't empty

        '''
        if len(self.list_of_states) == 0:
            return None, False

        poses = [state_dict['pose'] for state_dict in self.list_of_states if state_dict['timestep'] == t]

        return poses, len(poses) > 0

    def get_start_lane_index(self):
        def get_center_lane_index(pose):
            """
            Return the index of the inbound lane POSE is closest to,
            and whether the beginning pose is in the center of 
            the intersection.

            First return value is None if pose isn't in the center 
            square of intersection.
            """
            if pose[0] >= 400 and pose[0] <= 500:
                if pose[1] >= 400 and pose[1] <= 500:
                    return 3
                if pose[1] >= 500 and pose[1] <= 600:
                    return 5
            elif pose[0] >= 500 and pose[0] <= 600:
                if pose[1] >= 400 and pose[1] <= 500:
                    return 1
                if pose[1] >= 500 and pose[1] <= 600:
                    return 7
            else:
                return None

        valid_states = self.get_valid_states()
        if valid_states is None or len(valid_states) == 0:
            return None, None

        begin_lane = valid_states[0]['lane']
        if begin_lane is None:
            x_new, y_new = self.get_smoothed_spline_points()
            if x_new is None or y_new is None:
                return None, None
            return get_center_lane_index((x_new[0], y_new[0])), True
        else:
            return begin_lane['lane_index'], False

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

        #NSEW -> ENWS
        if lane['lane_index'] == 3:
            return np.pi/2

        elif lane['lane_index'] == 7:
            return -np.pi/2

        elif lane['lane_index'] == 1:
            return np.pi

        elif lane['lane_index'] == 5:
            return 0.0
        else: 
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

    def get_first_timestep(self):
        return min(self.list_of_states, key=lambda x: x['timestep'])['timestep']

    def get_last_timestep(self):
        return max(self.list_of_states, key=lambda x: x['timestep'])['timestep']


    def append_to_trajectory(self,datum):
        '''
        Append new pose to trajectory and updates the past angle to have
        the angle computed in compute_probability
        '''
        self.probability_list.append(self.prob_proposal)
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


    def compute_probability(self, state):
        '''
        Compute the log probability of the proposed state corresponding to this
        trajecotry 

        Returns
        ------------
        float, range [-inf,0] where 0 corrresponds to more probable 
        '''

        pos = self.return_last_state_pos()
        self.curr_state = [state['pose'][0], state['pose'][1]]

        curr_angle = self.compute_new_angle()
        valid_turn = is_valid_lane_change(self.list_of_states[-1]['lane'], state['lane'])
        valid_turn *= 1
        
        mean = np.array([pos[0], pos[1], self.past_angle, 1])
        state_full = np.array([self.curr_state[0], self.curr_state[1], curr_angle, valid_turn])

        var = measure_probability(self.cov, mean, state_full)

        self.prob_proposal = var
       
        self.curr_angle = curr_angle
        
        return var


    def prune_points_near_edge(self):
        indices_to_keep = []
        for i,state in enumerate(self.list_of_states):
            if not state['is_near_edge']:
                indices_to_keep.append(i)

        self.list_of_states = np.array(self.list_of_states)[indices_to_keep].tolist()

    def prune_points_outside_crosswalks(self):

        def on_uds_crosswalk(pose):

            def in_bound(n, bound):
                return n >= min(bound) and n <= max(bound)

            return in_bound(pose[0], (380, 620)) and in_bound(pose[1], (380, 620))\
                and not (in_bound(pose[0], (420, 580)) and in_bound(pose[1], (420, 580)))

        indices_to_keep = []
        for i,state in enumerate(self.list_of_states):
            if on_uds_crosswalk(state['pose']):
                indices_to_keep.append(i)

        self.list_of_states = np.array(self.list_of_states)[indices_to_keep].tolist()

        

    def fit_to_spline(self):
        valid_poses = np.array([state_dict['pose'] for state_dict in sorted(self.get_valid_states(), key=lambda x: x['timestep'])])
        # unique_poses = []
        # for x in sorted(np.unique(poses[:, 0])):
        #     unique_poses.append((x, np.average(poses[np.where(poses[:, 0]==x)][0][1])))

        self.xs = [x for x, y in valid_poses]
        self.ys = [y for x, y in valid_poses]
        if len(valid_poses) == 0:
            return None, None
        try:
            tck_and_u, fp, _, _ = splprep(np.array(valid_poses).T, s=40000, full_output=1)
            self.tck, self.u = tck_and_u
        except:
            print 'Caught error when fitting spline'
            return None, None
        return self.tck, self.u

    def get_smoothed_spline_points(self, sigma=75):
        tck, u = self.fit_to_spline()
        if tck is None or u is None:
            return None, None
        u_new = np.linspace(self.u.min(), self.u.max(), 1000)
        x_new, y_new = splev(u_new, self.tck)
        x_new = gaussian_filter1d(x_new, sigma)
        y_new = gaussian_filter1d(y_new, sigma)
        return x_new, y_new
