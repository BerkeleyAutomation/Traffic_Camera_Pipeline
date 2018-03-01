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

    def __init__(self, config):
        self.config = config

        self.car_crop_image = cv2.imread(self.config.car_crop_image_path)
        if self.car_crop_image is None:
            print 'Car crop image not found'
        else:
            self.car_crop_color = self.car_crop_image[0,0,:]

        self.pedestrian_crop_image = cv2.imread(self.config.pedestrian_crop_image_path)
        if self.pedestrian_crop_image is None:
            print 'Pedestrian crop image not found'
        else:
            self.pedestrian_crop_color = self.pedestrian_crop_image[0,0,:]


    def check_is_valid(self, rclass, x_min, y_min, x_max, y_max):
        rclass = int(rclass)
        x_min = int(self.config.img_dim[0] * x_min)
        y_min = int(self.config.img_dim[1] * y_min)
        
        x_max = int(self.config.img_dim[0] * x_max)
        y_max = int(self.config.img_dim[1] * y_max)

        # import pdb; pdb.set_trace()
        crop_mask = None
        full_mask = None
        # 6: bus, 7: car, 14: motorcycle
        if rclass in [6, 7, 14]:
            if self.car_crop_image is None:
                return True
            crop_mask = self.car_crop_image[y_min : y_max, x_min : x_max, :]
            full_mask = np.zeros(crop_mask.shape) + self.car_crop_color
        # 2: bicyle, 15: person
        elif rclass in [15]:
            if self.pedestrian_crop_image is None:
                return True
            crop_mask = self.pedestrian_crop_image[y_min : y_max, x_min : x_max, :]
            full_mask = np.zeros(crop_mask.shape) + self.pedestrian_crop_color
        else:
            return False

        assert crop_mask is not None
        assert full_mask is not None
        num_dims = full_mask.shape[0] * full_mask.shape[1] * full_mask.shape[2]
        num_match = np.count_nonzero(full_mask - crop_mask)
        ratio = float(num_match) / float(num_dims)

        if ratio > 0.5:
            return True
        else:
            return False
