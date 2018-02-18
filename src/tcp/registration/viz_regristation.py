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




class VizRegristration():


    def __init__(self,cnfg):

        self.config = cnfg


    def load_frames(self):
        self.imgs = []
        for i in range(500):
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
                                  visualizer=vis,
                                  max_time=500,
                                  randomize=False,
                                  agent_mappings={Car:NullAgent,
                                                  TrafficLight:TrafficLightAgent},
                                  use_ray=False
        )



	def visualize_trajectory_dots(self,trajectories):
		self.initalize_simulator(0,0)
        IPython.embed()
        print "got here"

        active_trajectories = []
		for t in range(500):

            for traj_index in range(len(trajectories)):

                traj = trajectories[traj_index]
                if t == traj.initial_time_step:

                    active_trajectories.append(traj)
                    del trajectories[traj_index]
                else: 
                    break

            way_points = []
            for traj_index in range(len(active_trajectories)):

                traj = active_trajectories[traj_index]
                next_state = traj.get_next_state()

                if next_state == None:
                    del active_trajectories[i]


            env._reset(new_state=init_state)
            env._render(waypoints = way_points)
            cv2.imshow('img',self.imgs[t])
            t+=1


        return
		



	def visualize_trajectory_cars(self,trajectory_of_objects)

		agent = RRTAgent()
		for i in range(20):
			trajectory, cls = get_single_trajectory(trajectories, i)

			if not len(trajectory):
				break

			tf_trajectory = transform_trajectory(trajectory)
			init_state.dynamic_objects[0] = Car(tf_trajectory[0][0], tf_trajectory[0][1], breadcrumbs=tf_trajectory)
			env._reset(new_state=init_state)
			env._render()

			state = env.current_state
			while(True):
				action = agent.eval_policy(state)
				state, reward, done, info_dict = env._step(action)
				env._render()
				if not state.dynamic_objects[0].breadcrumbs:
				    break
		return

