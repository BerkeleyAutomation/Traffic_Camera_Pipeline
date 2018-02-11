from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

#from AbstractDetector import AbstractDetector
from tcp.object_detection.video_labeler import LabelVideo
from tcp.registration.homography import Homography
from tcp.registration.iterative_filtering import IterativeFiltering
from tcp.registration.viz_regristration import VizRegristration
from tcp.object_detection.ssd_detector import SSD_VGG16Detector
from tcp.configs.alberta_config import Config
import IPython
import glob
import cPickle as pickle

VIDEO_FILE = 'Train_Videos/*.mp4'

cnfg = Config()
vl = LabelVideo(cnfg)
vr = VizRegristration(cnfg)
hm = Homography(cnfg,vz_debug=vr)


vr.load_frames()



iterative_filter = IterativeFiltering(cnfg)

camera_view_trajectories = []
camera_view_trajectory = pickle.load(open('test.p','r'))

camera_view_trajectories.append(camera_view_trajectory)

#hm.test_camera_point(camera_view_trajectory)


####GENERATE HOMOGRAPHY
simulator_view_trajectories = []
for camera_view_trajectory in camera_view_trajectories:
	simulator_view_trajectory = hm.transform_trajectory(camera_view_trajectory)
	filtered_trajectory = iterative_filter.heuristic_label(simulator_view_trajectory)
	


vr.visualize_trajectory_dots(filtered_trajectory)


IPython.embed()
