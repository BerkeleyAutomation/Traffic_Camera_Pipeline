#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector 
from tcp.object_detection.cropper import Cropper

import IPython

from tcp.object_detection.init_labeler import InitLabeler


class HardCodeLabel():

    def __init__(self,config,net_path =None):

        self.config = config



    def determine_label(self,obj):
        
        x,y,class_label,t = obj


        if t == 1:
            return (x,y,class_label,t,True)
        elif np.abs(x-0.953) < 0.01 and t == 107:
            return (x,y,class_label,t,True)
        elif t == 293:
            return (x,y,class_label,t,True)
        else:
            return (x,y,class_label,t,False)


    def label_video(self,video):
        new_video = []
        for frame in video:
            new_frame = []
            for obj in frame: 
                new_frame.append(self.determine_label(obj))

            new_video.append(new_frame)

        return new_video

if __name__ == "__main__":
    # main()
    annotateImage('../uds_video_demo/alberta_nobox.png')