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

###LABEL VIDEOS
for video_path in sorted(videos):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    datestamp = video_name.split('_')[-2]
    timestamp = video_name.split('_')[-1]

    year, month, date = [int(i) for i in datestamp.split('-')]
    hour, minute, second = [int(i) for i in timestamp.split('-')]

    # Setting first video
    tmp_time = int('%02d%02d%02d' % (date, hour, minute))
    if tmp_time < 270923:
        continue
    # Setting last video
    if tmp_time > 270923:
        break

    print 'Analyzing video: %s' % video_path

    camera_view_trajectory_pickle = '{0}/{1}/{1}_trajectories.cpkl'.format(cnfg.save_debug_pickles_path, video_name)
    camera_view_trajectory = pickle.load(open(camera_view_trajectory_pickle,'r'))

    assert camera_view_trajectory is not None, "%s doesn't have a trajectories pickle file" % video_name

    simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
    filtered_trajectory = of.heuristic_label(simulator_view_trajectory)
    
    for traj in filtered_trajectory:
        print ta.get_trajectory_primitive(traj)
        ta.visualize_trajectory(traj)
    
    # raw_input('\nPress enter to continue...\n')
