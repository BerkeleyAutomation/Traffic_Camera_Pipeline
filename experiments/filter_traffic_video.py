from __future__ import unicode_literals
import sys, os
import cv2

import numpy as np

#from AbstractDetector import AbstractDetector
from tcp.registration.homography import Homography
from tcp.registration.obs_filtering import ObsFiltering
from tcp.registration.viz_registration import VizRegistration
from tcp.configs.alberta_config import Config
import IPython
import glob
import cPickle as pickle




cnfg = Config()
vr = VizRegistration(cnfg)
hm = Homography(cnfg)
of = ObsFiltering(cnfg)

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
    if tmp_time < 270900:
        continue
    # Setting last video
    if tmp_time > 270905:
        break

    print 'Filtering video: %s' % video_path

    camera_view_trajectory_pickle = '{0}/{1}/{1}_trajectories.cpkl'.format(cnfg.save_debug_pickles_path, video_name)
    camera_view_trajectory = pickle.load(open(camera_view_trajectory_pickle,'r'))

    assert camera_view_trajectory is not None, "%s doesn't have a trajectories pickle file" % video_name

    simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
    filtered_trajectory = of.heuristic_label(simulator_view_trajectory)
    # vr.visualize_trajectory_dots(filtered_trajectory, filter_class='pedestrian', plot_traffic_images=False, video_name=video_name, animate=False)

    for traj in filtered_trajectory:
        if traj.class_label != 'pedestrian':
            continue
        # print traj.list_of_states
        vr.visualize_trajectory_dots([traj], filter_class='pedestrian', plot_traffic_images=False, video_name=video_name, animate=False)
        raw_input('press enter')

    raw_input('\nPress enter to continue...\n')
