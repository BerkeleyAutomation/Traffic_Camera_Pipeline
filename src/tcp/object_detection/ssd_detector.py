import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import math, random
import cv2
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

sys.path.insert(0, 'src/tcp/object_detection/SSD/')
from SSD.nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

from AbstractDetector import AbstractDetector

class SSD_VGG16Detector(AbstractDetector):
    def __init__(self, architechture, ckpt_path, stream_url=None):
        super(SSD_VGG16Detector, self).__init__(architechture, stream_url=stream_url)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        # Input placeholder.
        net_shape = (300, 300)
        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        image_pre, labels_pre, bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(image_pre, 0)

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        # Restore SSD model.
        ckpt_filepath = ckpt_path
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filepath)

        # SSD default anchor boxes.
        self.ssd_anchors = ssd_net.anchors(net_shape)

    """
        Main image processing routine.
    """
    def getBoundingBox(self, img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})
        
        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
                rpredictions, rlocalisations, self.ssd_anchors,
                select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
        
        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        # Original SSD output is (ymin, xmin, ymax, xmax)
        for i, rbbox in enumerate(rbboxes):
            rbboxes[i] = (rbbox[1], rbbox[0], rbbox[3], rbbox[2])
        return rclasses, rscores, rbboxes

    def drawBoundingBox(self, img, rclasses, rscores, rbboxes):
        visualization.bboxes_draw_on_img(img, rclasses, rscores, np.array(rbboxes), visualization.colors_plasma)
        return img
