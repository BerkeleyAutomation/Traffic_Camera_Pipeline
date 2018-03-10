import numpy as np

import pickle


#NSEW
initial_state_distribution = [0.25,0.25,0.25,0.25]


#NSEW with primtives order #FORWARD/RIGHT/LEFT
goal_state_distribution = [[0.25,0.25,0.25], #N
							[0.25,0.25,0.25], #S
							[0.25,0.25,0.25],#E
							[0.25,0.25,0.25]]#W


#Probability of a new car given current cars
temporal_distribution = [0.2,0.3,1.0/100,1.0/100,1.0/300,1.0/400,1.0/800,1.0/1800]


distributions = {}

distributions['initial_state'] = initial_state_distribution

distributions['goal_state'] = goal_state_distribution

distributions['new_car'] = temporal_distribution


pickle.dump(distributions,open('fake_uds_distributions.p','wb'))

