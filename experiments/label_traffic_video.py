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
from tcp.object_detection.ssd_detector import SSD_VGG16Detector
from tcp.configs.alberta_config import Config
import IPython
import cPickle as pickle
import glob

VIDEO_FILE = 'Train_Videos/*.mp4'

cnfg = Config()
vl = LabelVideo(cnfg)
hm = Homography(cnfg)
iterative_filter = IterativeFiltering(cnfg)


###GET VIDEOS
videos = glob.glob(VIDEO_FILE)



###LABEL VIDEOS
camera_view_trajectories = []
# IPython.embed()
# for video in videos[0]:
# 	IPython.embed()
# 	camera_view_trajectory = vl.label_video(video)
# 	camera_view_trajectories.append(camera_view_trajectory)

camera_view_trajectory = vl.label_video(videos[0], output_limit = 1000, debug_pickle = True)

pickle.dump(camera_view_trajectory,open('test_hard.p','wb'))



