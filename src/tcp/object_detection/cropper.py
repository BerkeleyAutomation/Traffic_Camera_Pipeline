#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector
import IPython



class Cropper():

    def __init__(self,config):

        self.config = config
        self.crop_image = cv2.imread(self.config.crop_image_path)
        self.crop_color = self.crop_image[0,0,:]


    def check_is_valid(self,x_min,x_max,y_min,y_max):

    	x_min = int(self.config.alberta_img_dim[0] * x_min)
    	x_max = int(self.config.alberta_img_dim[0] * x_max)

    	y_min = int(self.config.alberta_img_dim[1] * y_min)
    	y_max = int(self.config.alberta_img_dim[1] * y_max)

    	crop_mask = self.crop_image[y_min:y_max,x_min:x_max,:]

    	full_mask = np.zeros(crop_mask.shape) + self.crop_color

    	num_dims = full_mask.shape[0]*full_mask.shape[1]*full_mask.shape[2]

    	num_match = np.count_nonzero(full_mask-crop_mask)

    	ratio = float(num_match)/float(num_dims)

    	if ratio > 0.5:
    		return True
    	else:
    		return False
            



if __name__ == "__main__":
    # main()
    annotateImage('../uds_video_demo/alberta_nobox.png')