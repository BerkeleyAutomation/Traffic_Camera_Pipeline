class Trajectory():


    def __init__(self, initial_time_step,class_label,list_of_states):


      self.initial_time_step = initial_time_step
      self.class_label = class_label

      self.list_of_states = list_of_states


    def get_next_state(self):

      if len(self.list_of_states) == 0:
        return None

      state = self.list_of_states.pop(0)

      state_xy = [state[0],state[1]]

      return state_xy