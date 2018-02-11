import gym
import gym_urbandriving as uds
import cProfile
import time
import numpy as np
import pickle
import skimage.transform
import cv2
import IPython

from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent, RRTAgent
from gym_urbandriving.assets import Car, TrafficLight

import colorlover as cl



class VizRegristration():

    def __init__(self,cnfg):

        self.config = cnfg


    def load_frames(self):
        self.imgs = []
        for i in range(1,500):
            img = cv2.imread(self.config.save_debug_img_path+'img_'+str(i)+'.png')
            self.imgs.append(img)

    def initalize_simulator(self,ncars,nped):

        self.vis = uds.PyGameVisualizer((800, 800))

        # Create a simple-intersection state, with 4 cars, no pedestrians, and traffic lights
        self.init_state = uds.state.SimpleIntersectionState(ncars=ncars, nped=nped, traffic_lights=True)

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


    def compute_color_template(self,num_trajectories):

        color_interp = np.linspace(0,255,num = num_trajectories)
        color_template = []

        for i in range(num_trajectories):

            color = int(color_interp[i])
            color_template.append((color,color,color))
       
        return color_template



    def visualize_trajectory_dots(self,trajectories):
        self.initalize_simulator(0,0)

        active_trajectories = []
        way_points = []

        color_template = self.compute_color_template(len(trajectories))

        color_index = 0
        for t in range(500):
            print "T ",t

            for traj_index in range(len(trajectories)):

                traj = trajectories[traj_index]
                
                if t == traj.initial_time_step:
                    
                    color_match = [traj,color_template[color_index]]
                    active_trajectories.append(color_match)
                    color_index += 1
            
            
            for traj_index in range(len(active_trajectories)):

                traj = active_trajectories[traj_index][0]
                next_state = traj.get_next_state()


                if not next_state == None:
                    w_p = [next_state,active_trajectories[traj_index][1]]
                    way_points.append(w_p)


            self.env._render(traffic_trajectories  = way_points)
           
            cv2.imshow('img',self.imgs[t])
            cv2.waitKey(30)
            t+=1


        return

    def visualize_homography_points(self):
        self.initalize_simulator(0,0)
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

    def visualize_simulator_point(self,x,y):
        self.initalize_simulator(0,0)
        
        waypoints = [[x,y]]
        
        self.env._render(waypoints = waypoints)     
        cv2.imshow('img',img)
        cv2.waitKey(30)

    def visualize_camera_point(self,x,y,t):


        img = self.imgs[t]

        point = [x,y]

        img[point[1]-5:point[1]+5,point[0]-5:point[0]+5,:]=255
        
        cv2.imshow('img',img)
        cv2.waitKey(30)

            
		



	# def visualize_trajectory_cars(self,trajectory_of_objects)

	# 	agent = RRTAgent()
	# 	for i in range(20):
	# 		trajectory, cls = get_single_trajectory(trajectories, i)

	# 		if not len(trajectory):
	# 			break

	# 		tf_trajectory = transform_trajectory(trajectory)
	# 		init_state.dynamic_objects[0] = Car(tf_trajectory[0][0], tf_trajectory[0][1], breadcrumbs=tf_trajectory)
	# 		env._reset(new_state=init_state)
	# 		env._render()

	# 		state = env.current_state
	# 		while(True):
	# 			action = agent.eval_policy(state)
	# 			state, reward, done, info_dict = env._step(action)
	# 			env._render()
	# 			if not state.dynamic_objects[0].breadcrumbs:
	# 			    break
	# 	return

