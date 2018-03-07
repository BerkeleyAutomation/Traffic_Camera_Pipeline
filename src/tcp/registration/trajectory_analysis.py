import sys, os
import cv2
import numpy as np
import cPickle as pickle

import matplotlib.pyplot as plt
import seaborn as sns

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
        valid_states = trajectory.get_valid_states()
        if len(valid_states) < 20:
            print 'Trajectory too short with length %d' % len(valid_states)
            return None
        
        tck, u = trajectory.fit_to_spline()
        assert tck is not None and u is not None

        x_new, y_new = trajectory.get_smoothed_spline_points()
        traj_len = len(x_new)

        x_new_diff = np.diff(x_new[: traj_len // 10])
        y_new_diff = np.diff(y_new[: traj_len // 10])
        # print x_new
        begin_pose = (x_new[0], y_new[0])
        begin_angle = np.degrees(np.arctan2(y_new_diff, x_new_diff))
        begin_angle = np.average(begin_angle)

        x_new_diff = np.diff(x_new[-(traj_len // 10) : -1])
        y_new_diff = np.diff(y_new[-(traj_len // 10) : -1])
        end_pose = (x_new[-1], y_new[-1])
        end_angle = np.degrees(np.arctan2(y_new_diff, x_new_diff))
        end_angle = np.average(end_angle)

        diff_angle = end_angle - begin_angle
        if diff_angle < -180 or diff_angle > 180:
            diff_angle %= 360

        dist = np.sqrt((end_pose[0] - begin_pose[0]) ** 2 + (end_pose[1] - begin_pose[1]) ** 2)

        if dist < 100:
            return 'stopped'

        begin_state = trajectory.get_states_at_timestep(trajectory.get_first_timestep())[0]
        end_state = trajectory.get_states_at_timestep(trajectory.get_last_timestep())[0]
        begin_lane = begin_state['lane']
        end_lane = end_state['lane']
        # print begin_lane, end_lane
        if begin_lane is not None and end_lane is not None:
            begin_lane_index = begin_lane['lane_index']
            end_lane_index = end_lane['lane_index']
            if begin_lane_index %2 != 0:
                if utils.FORWARD_LANE_CHANGE[begin_lane_index] == end_lane_index:
                    return 'forward'
                elif utils.LEFT_TURN_LANE_CHANGE[begin_lane_index] == end_lane_index:
                    return 'left'
                elif utils.RIGHT_TURN_LANE_CHANGE[begin_lane_index] == end_lane_index:
                    return 'right'

        # print 'pose: %s, %s' % (str(begin_pose), str(end_pose))
        # print 'angles: %.2f - %.2f = %.2f' % (end_angle, begin_angle, diff_angle)
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

        if tck is None and u is None:
            return

        plt.figure(figsize=(10, 10))
        axes = plt.gca()
        axes.set_xlim([-100,1100])
        axes.set_ylim([-100,1100])
        plt.gca().invert_yaxis()

        plt.plot([400, 400], [0, 1000], color='k')
        plt.plot([500, 500], [0, 1000], color='k')
        plt.plot([600, 600], [0, 1000], color='k')

        plt.plot([0, 1000], [400, 400], color='k')
        plt.plot([0, 1000], [500, 500], color='k')
        plt.plot([0, 1000], [600, 600], color='k')

        plt.scatter(trajectory.xs, trajectory.ys, c='r', marker='.')

        x_new, y_new = trajectory.get_smoothed_spline_points()
        plt.plot(x_new, y_new)
        plt.show()