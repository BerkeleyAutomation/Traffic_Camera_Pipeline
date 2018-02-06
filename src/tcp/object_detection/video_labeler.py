#/home/jren/env/bin/python
from __future__ import unicode_literals

import sys, os
import cv2
import youtube_dl
import numpy as np
from urlparse import parse_qs

from AbstractDetector import AbstractDetector
from ssd_detector import SSD_VGG16Detector



class LabelVideo():

    def __init__(self,config,net_path =None):

        self.ssd_detector = SSD_VGG16Detector('ssd_vgg16', config.check_point_path)



    def label_video(self,video_name, output_limit=1000, num_skip_frames=1):
            ssd_detector.setStreamURL(os.path.join(VIDEO_ROOT_DIR, video_name))
            if not AbstractDetector.openCapture(ssd_detector):
                raise ValueError('Video file %s failed to open.' % video_name)
            print 'Scanning %s...' % (video_name)

            output_count = 0
            frame_skip = 0
            while ssd_detector.cap.isOpened():
                ret, frame = ssd_detector.cap.read()
                if frame is None:
                    break

                frame_skip += 1
                assert num_skip_frames > 0 and int(num_skip_frames) == num_skip_frames
                frame_skip %= num_skip_frames
                if frame_skip != 0:
                    continue

                # Process frame here
                rclasses, rscores, rbboxes = ssd_detector.process_image(frame)
                unique, counts = np.unique(rclasses, return_counts=True)
                classes_counts = dict(zip(unique, counts))
                car_count = classes_counts.get(7)
                if car_count is not None and car_count > 0:
                    image_name = '%s_%d.png' % (video_name[:-4], output_count)
                    IPython.embed()

            print 'Writen %d images.' % (output_count)
            ssd_detector.cap.release()


    def make_a_new_point(self,input):

        return



if __name__ == "__main__":
    # main()
    annotateImage('../uds_video_demo/alberta_nobox.png')