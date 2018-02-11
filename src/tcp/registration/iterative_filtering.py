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
from tcp.registration.trajectory import Trajectory



class IterativeFiltering():

    def __init__(self,config):

        self.config = config
        

    def ang(self,v1, v2):
        # Returns the angle in radians between vectors 'v1' and 'v2'
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def heuristic_label(self,trajectories, dist_thresh=100, t_thresh=5):
        traj_holder = []
        
        for frame in trajectories:

            for x, y, cls, t in frame:
                
                possible_paths = []
                for i, (foundcls, traj) in enumerate(traj_holder):
                    if foundcls == cls:
                        lastx, lasty, lastt = traj[-1]
                        dist = np.linalg.norm([x-lastx, y-lasty])
                        print "DIST ",dist
                        if dist < dist_thresh and t - lastt < t_thresh and lastt != t:
                            possible_paths.append(i)
                            print "thought possible"

                if not len(possible_paths):
                    traj_holder.append((cls, [(x, y, t)]))

                else:
                    best_angle = 999
                    best_possible = -1
                    for i in possible_paths:
                        foundcls, traj = traj_holder[i]

                        if len(traj) == 1:
                            best_angle = 0
                            best_possible = i

                        else:
                            l_coord, ll_coord = traj[-1], traj[-2]
                            lx, ly, _ = l_coord
                            llx, lly, _ = ll_coord
                            lv = lx - llx, ly - lly
                            cv = x - lx, y - ly
                            angle = self.ang(lv, cv)
                            if angle < best_angle:
                                best_angle = angle
                                best_possible = i

                    traj_holder[best_possible][1].append((x, y, t))



        new_trajectories = []

       
        for traj in traj_holder:
            
            initial_time_step = traj[1][0][2]
            clss_label = traj[0]
            new_trajectories.append(Trajectory(initial_time_step,clss_label,traj[1]))

        
        return new_trajectories

