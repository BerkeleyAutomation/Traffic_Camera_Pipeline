import sys, os
import cv2
import numpy as np
import cPickle as pickle

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.interpolate import splev

import gym
import gym_urbandriving as uds
from gym_urbandriving.agents import KeyboardAgent, AccelAgent, NullAgent, TrafficLightAgent
from gym_urbandriving.assets import Car, TrafficLight

import tcp.utils.utils as utils
from tcp.registration.trajectory import Trajectory


class TrajectoryAnalysis:

    def __init__(self, config):
        self.config = config

    """
        Set of primitives include:
            - forward: "forward"
            - left turn: "left"
            - right turn: "right"
            - U-turn: "u-turn"
            - stopped: "stop"
            - uncertain: None
    """
    def get_trajectory_primitive(self, trajectory):
        tck, u = trajectory.fit_to_spline()
        
        u_new = np.linspace(u[0], u[10], 50)
        x_new, y_new = splev(u_new, tck)
        x_new_diff = np.diff(x_new)
        y_new_diff = np.diff(y_new)
        begin_angle = np.degrees(np.arctan2(y_new_diff, x_new_diff))
        begin_angle = np.average(begin_angle)

        u_new = np.linspace(u[-10], u[-1], 50)
        x_new, y_new = splev(u_new, tck)
        x_new_diff = np.diff(x_new)
        y_new_diff = np.diff(y_new)
        end_angle = np.degrees(np.arctan2(y_new_diff, x_new_diff))
        end_angle = np.average(end_angle)

        diff_angle = end_angle - begin_angle

        begin_pose = np.average(trajectory.get_poses_at_timestep(trajectory.get_first_timestep())[0], axis=0)
        end_pose = np.average(trajectory.get_poses_at_timestep(trajectory.get_last_timestep())[0], axis=0)

        dist = np.sqrt((end_pose[0] - begin_pose[0]) ** 2 + (end_pose[1] - begin_pose[1]) ** 2)

        if dist < 50:
            return 'stopped'


        begin_state = trajectory.get_states_at_timestep(trajectory.get_first_timestep())[0]
        end_state = trajectory.get_states_at_timestep(trajectory.get_last_timestep())[0]
        begin_lane = begin_state['lane']
        end_lane = end_state['lane']
        if begin_lane is not None and end_lane is not None:
            begin_lane_index = begin_lane['lane_index']
            end_lane_index = end_lane['lane_index']
            if utils.FORWARD_LANE_CHANGE[begin_lane_index] == end_lane_index:
                return 'forward'
            elif utils.LEFT_TURN_LANE_CHANGE[begin_lane_index] == end_lane_index:
                return 'left'
            elif utils.RIGHT_TURN_LANE_CHANGE[begin_lane_index] == end_lane_index:
                return 'right'

        if diff_angle >= -120 and diff_angle <= -60:
            return 'left'
        elif diff_angle >= 60 and diff_angle <= 120:
            return 'right'
        elif diff_angle >= -30 and diff_angle <= 30:
            return 'forward'
        elif diff_angle <= -165 or diff_angle >= 165:
            return 'u-turn'


    def visualize_trajectory(self, trajectory):
        tck, u = trajectory.fit_to_spline()

        plt.figure(figsize=(6.5, 4))
        axes = plt.gca()
        axes.set_xlim([-100,1100])
        axes.set_ylim([-100,1100])
        plt.gca().invert_yaxis()
        plt.scatter(trajectory.xs, trajectory.ys, c='r', marker='.')

        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck)
        plt.plot(x_new, y_new)
        plt.show()