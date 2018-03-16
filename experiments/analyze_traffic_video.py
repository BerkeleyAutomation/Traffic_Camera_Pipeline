from __future__ import unicode_literals

import sys, os
import cv2

import numpy as np

from tcp.registration.homography import Homography
from tcp.registration.obs_filtering import ObsFiltering
from tcp.registration.viz_registration import VizRegistration
from tcp.registration.trajectory_analysis import TrajectoryAnalysis
from tcp.configs.alberta_config import Config

import glob
import cPickle as pickle

cnfg = Config()
vr = VizRegistration(cnfg)
hm = Homography(cnfg)
of = ObsFiltering(cnfg)
ta = TrajectoryAnalysis(cnfg)

###GET VIDEOS
VIDEO_FILE = '%s/*.mp4' % cnfg.video_root_dir
videos = glob.glob(VIDEO_FILE)

### Analyze Videos
car_stat_dict = {
        1: {'left': 0, 'forward': 0, 'right': 0, 'stopped': 0},
        3: {'left': 0, 'forward': 0, 'right': 0, 'stopped': 0},
        5: {'left': 0, 'forward': 0, 'right': 0, 'stopped': 0},
        7: {'left': 0, 'forward': 0, 'right': 0, 'stopped': 0}
    }

pedestrian_stat_dict = {
    'north_clw': 0,
    'north_ccw': 0,
    'south_clw': 0,
    'south_ccw': 0,
    'west_clw': 0,
    'west_ccw': 0,
    'east_clw': 0,
    'east_ccw': 0
}

for video_path in sorted(videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    datestamp = video_name.split('_')[-2]
    timestamp = video_name.split('_')[-1]

    year, month, date = [int(i) for i in datestamp.split('-')]
    hour, minute, second = [int(i) for i in timestamp.split('-')]

    # Setting first video
    tmp_time = int('%02d%02d%02d' % (date, hour, minute))
    if tmp_time < 270900:
        continue
    # Setting last video
    if tmp_time > 271300:
        break

    print 'Analyzing video: %s' % video_path

    camera_view_trajectory_pickle = '{0}/{1}/{1}_trajectories.cpkl'.format(cnfg.save_debug_pickles_path, video_name)
    camera_view_trajectory = pickle.load(open(camera_view_trajectory_pickle,'r'))

    assert camera_view_trajectory is not None, "%s doesn't have a trajectories pickle file" % video_name

    simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
    filtered_trajectory = of.heuristic_label(simulator_view_trajectory)
    
    for i, traj in enumerate(filtered_trajectory):
        if traj.class_label != 'pedestrian':
            continue
        # start_lane_index, _ = traj.get_start_lane_index()
        # primitive = ta.get_trajectory_primitive(traj)

        # if start_lane_index is not None and\
        #     car_stat_dict.get(start_lane_index) is not None:

        #     if primitive is not None and\
        #         car_stat_dict[start_lane_index].get(primitive) is not None:
        #         car_stat_dict[start_lane_index][primitive] += 1


        # ta.save_trajectory(traj, video_name, i)
        traj.prune_points_outside_crosswalks()
        pedestrian_primitive = ta.get_pedestrian_trajectory_primitive(traj)
        if pedestrian_stat_dict.get(pedestrian_primitive) is not None:
            pedestrian_stat_dict[pedestrian_primitive] += 1

        # ta.visualize_trajectory(traj)

    # raw_input('\nPress enter to continue...\n')

# for key in car_stat_dict:
#     print 'begin from lane %d: ' % key, car_stat_dict[key]

# with open(os.path.join(cnfg.save_debug_pickles_path, 'primitive_count_dict.pkl'), 'w+') as pkl_file:
#     pickle.dump(car_stat_dict, pkl_file)
