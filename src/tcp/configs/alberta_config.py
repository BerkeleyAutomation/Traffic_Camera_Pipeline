import os
import numpy as np
#
# path and dataset parameter
#


class Config(object):
###############PARAMETERS TO SWEEP##########


	def __init__(self):
		# Stream Capture
		self.STREAM_OUTPUT_SEGMENT_TIME_LIMIT = 60 	# approximate length of video segments in seconds
		self.STREAM_OUTPUT_DIR_SIZE_LIMIT = 1e9 	# maximum size limit for downloaded video in bytes

		# Object Detection
		self.check_point_path = 'Checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

		# Registration

