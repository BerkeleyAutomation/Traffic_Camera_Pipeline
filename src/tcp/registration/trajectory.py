import numpy as np
from scipy.stats import multivariate_normal
import copy
import IPython

from tcp.utils.utils import compute_angle,measure_probability
class Trajectory():


    def __init__(self, obj):


      self.initial_time_step = obj['timestep']
      self.class_label = obj['class_label']
      self.list_of_states = []
      self.index = -1
      self.proposal_states = [obj]
      self.still_on = True

      self.cov = np.array([[ 6.96906626,  1.93169669, -0.1744157 ],
                           [ 1.93169669,  6.26641719, -0.55359298],
                           [-0.1744157,  -0.55359298,  1.91510572]])

      # self.cov = np.array([[ 0.396906626,  1.93169669, -0.1744157 ],
      #                      [ 1.93169669,  7.46641719, -0.55359298],
      #                      [-0.1744157,  -0.55359298,  3.391510572]])


    def get_next_state(self):

      if len(self.list_of_states) == 0:
        return None,False

      state = self.list_of_states.pop(0)

      return state['pose'],True


    def return_last_state_pos(self):

        state = self.list_of_states[-1]
        return state['pose']

    def return_proposal_state_pos(self,indx):
        
        state = self.proposal_states[indx]
        return state['pose']

    def return_first_state_pos(self):

        state = self.list_of_states[0]
        return state['pose']



    def append_to_trajectory(self,datum):

        self.list_of_states.append(datum)


    def lane_matches_end_point(self,obj):

        end_lane = self.list_of_states[-1]['lane']

        if end_lane == None:
          return True

        lane = obj['lane']

        if (['lane_index'] == end_lane['lane_index']):
          if (lane['road_side'] == end_lane['road_side']):
            return True

        return False

    def compute_current_angle(self,indx):

        if len(self.list_of_states) < 1: 

          lane = self.proposal_states[indx]['lane']

          if lane['lane_index'] == 0:
            return np.pi/2

          elif lane['lane_index'] == 1:
            return -np.pi/2

          elif lane['lane_index'] == 2:
            return np.pi

          elif lane['lane_index'] == 3:
            return 0.0

        else:

          pos = self.return_proposal_state_pos(indx)
          pos_b = self.return_last_state_pos()

          angle = compute_angle(pos,pos_b)

          return angle

    def select_proposal_state(self,index):

        datum = copy.deepcopy(self.proposal_states[index])
       
        self.list_of_states.append(datum)

    def add_proposal_state(self,datum):

        last_state = self.list_of_states[-1]

        if datum['timestep'] - last_state['timestep'] > 50:
          self.still_on = False
          return
       
        self.proposal_states.append(datum)

    def clear_proposal_states(self):
        self.proposal_states = []

    def compute_new_angle(self,state,indx):

        pos = self.return_proposal_state_pos(indx)

        vector = state['pose'][0] - pos[0],state['pose'][1]-pos[1]

        angle = compute_angle(state['pose'],pos)

        return angle

    




    def compute_probability(self,state):

        ####ADD IN LENGHT AND CURRENT ANGLE

        best_prob = -1000000000
        best_prob_indx = -1
        for i in range(len(self.proposal_states)):
          angle = self.compute_current_angle(i)

          if len(self.list_of_states) < 1:
            pos = self.return_proposal_state_pos(i)
          else:
            pos = self.return_last_state_pos()

          mean = np.array([pos[0],pos[1],angle])

          curr_angle = self.compute_new_angle(state,i)
          state_full = np.array([state['pose'][0],state['pose'][1],curr_angle])

         
          var = measure_probability(self.cov,mean,state_full)


          if len(self.list_of_states) < 1: 
            prob_proposal =  var
          else:    
            prob_proposal =  0.001*var


          if prob_proposal > best_prob:
            best_prob = prob_proposal
            best_prob_indx = i

        return best_prob,best_prob_indx





       