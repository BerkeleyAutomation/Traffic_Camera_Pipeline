import numpy as np
import IPython


def compute_angle(pos_2,pos_1):

	'''
	Compute the angle between two vectors. pos_2 should follow pos_1 temporally

	Parameters
	----------
	pos2: np.array([x,y])
	pos1: np.array([x,y])

	Return
	---------
	float: corresponding to anlge
	'''


	difference = pos_2[0]-pos_1[0] , pos_2[1] - pos_1[1]

	angle = np.arctan2(difference[1],difference[0])

	return angle

def euclidean_angle_distance(angle_1,angle_2):
	'''
	Measure the distance between two angles. Note this is Euclidean distance, so 
	-pi and pi would have a distance of zero

	Parameters
	------------
	angle_1: float
		range [-pi,pi]
	angle_2: float 
		range [-pi,pi]

	Return
	------------
	float: euclidean angle distance

	'''

	return np.abs((angle_1+np.pi - angle_2)%(2*np.pi)-np.pi)



def measure_probability(cov,mean,state):

	''''
	Measure the unnormalized Gaussian probalility of a new state occuring 

	Parameters
	------------
	cov: np.array
		covaraince matrix with shape 3,3
	mean: np.array([x,y,angle])
	state: np.array([x,y,angle])

	Returns 
	-----------
	float, log probability of value 
	'''



	x_delta = mean[0] - state[0]
	y_delta = mean[1] - state[1]
	angle_delta = euclidean_angle_distance(mean[2],state[2])

	dif_state = np.array([[x_delta,y_delta,angle_delta]])

	prob = -np.dot(dif_state,np.dot(cov,dif_state.T))
	
	return prob[0,0]


######CRUDE TEST CASES##########

if __name__ == "__main__":

	pos_1 = [613,466]

	pos_2 = [612,466]

	print "COMPUTE ANGLE ",compute_angle(pos_2,pos_1)


	print "EUCLIDEAN DISTANCE ",euclidean_angle_distance(np.pi,-np.pi)