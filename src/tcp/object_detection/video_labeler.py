#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os, pdb
import time
import cv2
import youtube_dl
import numpy as np
import cPickle as pickle
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector 
from tcp.object_detection.cropper import Cropper

import IPython

from tcp.object_detection.init_labeler import InitLabeler
from tcp.object_detection.init_labeler_opencv import InitLabeler_OpenCV
from tcp.object_detection.visualization import colors_tableau, bboxes_draw_on_img


class LabelVideo():

    def __init__(self, config, net_path=None):

        self.config = config
        self.cropper = Cropper(self.config)

        self.ssd_detector = SSD_VGG16Detector('ssd_vgg16', self.config.check_point_path, cropper=self.cropper)

    def label_video(self, video_path, output_limit=None, num_skip_frames=1, debug_pickle=False):
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        self.ssd_detector.setStreamURL(video_path)
        if not AbstractDetector.openCapture(self.ssd_detector):
            raise ValueError('Video file %s failed to open.' % video_path)
        print 'Scanning %s...' % (video_path)


        ### First pass: get bounding ###
        all_rclasses = []
        all_rbboxes = []

        if debug_pickle:
            try:
                all_rclasses = pickle.load(open('{0}/{1}/{1}_classes.cpkl'.format(self.config.save_debug_pickles_path, video_name), 'r'))
                print 'Loaded "{0}/{1}/{1}_classes.cpkl".'.format(self.config.save_debug_pickles_path, video_name)
            except IOError as e:
                print 'Unable to load "{0}/{1}/{1}_classes.cpkl"'.format(self.config.save_debug_pickles_path, video_name)
            try:
                all_rbboxes = pickle.load(open('{0}/{1}/{1}_bboxes.cpkl'.format(self.config.save_debug_pickles_path, video_name), 'r'))
                print 'Loaded "{0}/{1}/{1}_bboxes.cpkl".'.format(self.config.save_debug_pickles_path, video_name)
            except IOError as e:
                print 'Unable to load "{0}/{1}/{1}_bboxes.cpkl"'.format(self.config.save_debug_pickles_path, video_name)

        if all_rclasses == [] or all_rbboxes == []:
            print 'Some detection pickle file failed to load. Running detector network... This may take a while.'
            while self.ssd_detector.cap.isOpened():
                ret, frame = self.ssd_detector.cap.read()
                if frame is None:
                    break
                rclasses, rscores, rbboxes = self.ssd_detector.get_bounding_box(frame)

                # Filter bboxes with cropper mask
                rclasses = [rclasses[i] for i, bbox in enumerate(rbboxes) if self.cropper.check_is_valid(*bbox)]
                rbboxes = [bbox for bbox in rbboxes if self.cropper.check_is_valid(*bbox)]

                all_rclasses.append(rclasses)
                all_rbboxes.append(rbboxes)
            self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if debug_pickle:
            if not os.path.exists('{0}/{1}'.format(self.config.save_debug_pickles_path, video_name)):
                os.makedirs('{0}/{1}'.format(self.config.save_debug_pickles_path, video_name))
            pickle.dump(all_rclasses, open('{0}/{1}/{1}_classes.cpkl'.format(self.config.save_debug_pickles_path, video_name), 'w+'))
            pickle.dump(all_rbboxes, open('{0}/{1}/{1}_bboxes.cpkl'.format(self.config.save_debug_pickles_path, video_name), 'w+'))

        ### CALL INITIAL LABELER ###
        start_time = time.time()
        self.init_labeler = InitLabeler_OpenCV(self.config, self.ssd_detector.cap, all_rbboxes, all_rclasses,
                                        video_name=video_name, cache_frames=True)
        elapsed_time = time.time() - start_time

        with open('{0}/{1}/{1}_timing.txt'.format(self.config.save_debug_pickles_path, video_name),'a+') as timing_file:
            timing_file.write('InitLabeler timing: %d min %d sec (%d seconds)\n\n' % (elapsed_time // 60, elapsed_time % 60, elapsed_time))

        self.all_rbboxes = self.init_labeler.all_rbboxes
        self.all_rclasses = self.init_labeler.all_rclasses

        frame_i = 0
        frame_skip = 0

        trajectory = []

        ### Second pass: process bounding box data ###
        self.ssd_detector.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self.ssd_detector.cap.isOpened():
            # pdb.set_trace()
            ret, frame = self.ssd_detector.cap.read()
            if frame is None:
                break

            frame_skip += 1
            
            assert num_skip_frames > 0 and int(num_skip_frames) == num_skip_frames
            frame_skip %= num_skip_frames
            if frame_skip != 0:
                continue

            # Process frame here
            rclasses = all_rclasses[frame_i]
            rbboxes = all_rbboxes[frame_i]

            if self.config.save_images:
                debug_img_path = os.path.join(self.config.save_debug_img_path, video_name)
                if not os.path.exists(debug_img_path):
                    os.makedirs(debug_img_path)
                bboxes_draw_on_img(frame, rclasses, [1.0] * len(rclasses), rbboxes, colors_tableau)
                cv2.imwrite(os.path.join(debug_img_path, '%s_%07d.jpg' % (video_name, frame_i)), frame)
            
            unique, counts = np.unique(rclasses, return_counts=True)
            classes_counts = dict(zip(unique, counts))
            car_count = classes_counts.get(7)

            if frame_i % 100 == 0:
                if output_limit is None:
                    total_frames = len(all_rclasses)
                else:
                    total_frames = min(output_limit, len(all_rclasses))
                print "Processed frames: %d/%d " % (frame_i, total_frames)

            current_frame = []

            car_cords = self.get_car_cords(frame_i, rclasses, rbboxes)
            ped_cords = self.get_pedestrian_cords(frame_i, rclasses, rbboxes)

            frame_i += 1
            if output_limit is not None and frame_i > output_limit:
                break

            for car_cord in car_cords:
                current_frame.append(car_cord)

            if self.config.use_pedestrian:
                for ped_cord in ped_cords:
                    current_frame.append(ped_cord)  

            if len(current_frame) != 0:
                trajectory.append(current_frame)

        print 'Done processing %d frames.' % (frame_i - 1)
        self.ssd_detector.cap.release()
        
        return trajectory


    def get_car_cords(self, frame_i, rclasses, rbboxes):
        rclasses = np.array(rclasses)
        rbboxes = np.array(rbboxes)
        assert len(rclasses) == len(rbboxes), 'rclasses: %s\n rbboxes: %s' % (rclasses, rbboxes)

        frame = []
        arg_init_label = self.init_labeler.get_arg_init_label(frame_i)
        for i in range(len(rclasses)):
            if rclasses[i] == 7:
                x_min, y_min, x_max, y_max = rbboxes[i]
                point = {'x': (x_min + x_max) / 2.0,
                         'y': y_max,
                         'cls_label': 'car',
                         't': frame_i}
                if i in arg_init_label:
                    point['is_initial_state'] = True
                else:
                    point['is_initial_state'] = False
                frame.append(point)
        return frame


    def get_pedestrian_cords(self, frame_i, rclasses, rbboxes):
        rclasses = np.array(rclasses)
        rbboxes = np.array(rbboxes)
        assert len(rclasses) == len(rbboxes), 'rclasses: %s\n rbboxes: %s' % (rclasses, rbboxes)

        frame = []
        arg_init_label = self.init_labeler.get_arg_init_label(frame_i)
        for i in range(len(rclasses)):
            if rclasses[i] == 15:
                x_min, y_min, x_max, y_max = rbboxes[i]
                point = {'x': (x_min + x_max) / 2.0,
                         'y': (y_min + y_max) / 2.0,
                         'cls_label': 'pedestrian',
                         't': frame_i}
                if i in arg_init_label:
                    point['is_initial_state'] = True
                else:
                    point['is_initial_state'] = False
                frame.append(point)

        return frame
