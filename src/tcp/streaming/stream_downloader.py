import os
import cv2
import numpy as np

from time import time, localtime, mktime, strftime

class StreamDownloader(object):
    FPS_CAP = 60

    """
        Initializes a stream downloader.
        :param stream_url: A String representing the stream URL. It can be a file path or .m3u8 URL for live streams.
    """
    def __init__(self, stream_src, stream_save_path, config):
        self.stream_src = stream_src
        self.stream_save_path = stream_save_path
        self.config = config
        self.frame_count = 0
        self.cap = None

    def check_capture(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            opened = self.open_capture()
            if not opened:
                raise ValueError('Stream cannot be opened.')

    """
        Open a cv2 capture object.
        :return: True if opening capture was successful, false otherwise.
        :rtype: Boolean
    """
    def open_capture(self):
        if self.stream_src is None:
            raise ValueError("Stream URL isn't defined. Please define stream_src in constructor.")

        self.cap = cv2.VideoCapture(self.stream_src)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.cap.isOpened()

    def display_video(self):
        self.check_capture()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps >= StreamDownloader.FPS_CAP:
            fps = StreamDownloader.FPS_CAP
        print 'Displaying with FPS = %d' % fps
        try:
            print 'Press Q to stop video stream.'
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if frame is None:
                    break
                
                cv2.imshow('Video Stream: "%s"' % (self.stream_src), frame)
                if cv2.waitKey(1000 / fps) & 0xFF == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    break
        except KeyboardInterrupt:
            self.cap.release()
            cv2.destroyAllWindows()

    def save_video(self, filename, save_frames=False):
        self.check_capture()

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps >= StreamDownloader.FPS_CAP:
            fps = StreamDownloader.FPS_CAP
        assert fps > 0, "FPS can't be negative."

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'X264')

        filepath = os.path.join(self.stream_save_path, filename)
        timestamp = localtime()
        timestr = strftime('%Y-%m-%d_%H-%M-%S', timestamp)
        out = cv2.VideoWriter('%s_%s.mp4' % (filepath, timestr), fourcc, fps, (self.width, self.height))
        
        dir_size_maxed = False
        try:
            while self.cap.isOpened():
                if self.config.STREAM_OUTPUT_SEGMENT_TIME_LIMIT is not None:
                    if time() - mktime(timestamp) > self.config.STREAM_OUTPUT_SEGMENT_TIME_LIMIT:
                        timestamp = localtime()
                        timestr = strftime('%Y-%m-%d_%H-%M-%S', timestamp)
                        out.open('%s_%s.mp4' % (filepath, timestr), fourcc, fps, (self.width, self.height))

                ret, frame = self.cap.read()
                if frame is None:
                    break

                out.write(frame)
                if save_frames:
                    output_frames_path = os.path.join(self.stream_save_path, 'frames')
                    if not os.path.exists(output_frames_path):
                        os.makedirs(output_frames_path)
                    output_frames_path = os.path.join(output_frames_path, '%s_%010d.jpg' % (filename, self.frame_count))
                    cv2.imwrite(output_frames_path, frame)
                self.frame_count += 1


                if self.config.STREAM_OUTPUT_DIR_SIZE_LIMIT is not None:
                    dir_size = sum(os.path.getsize(os.path.join(self.stream_save_path, f))\
                                   for f in os.listdir(self.stream_save_path)\
                                   if os.path.isfile(os.path.join(self.stream_save_path, f)))
                    if dir_size > self.config.STREAM_OUTPUT_DIR_SIZE_LIMIT:
                        dir_size_maxed = True
                        break

            out.release()
            self.cap.release()
            return dir_size_maxed
        except KeyboardInterrupt:
            out.release()
            self.cap.release()
            return dir_size_maxed
