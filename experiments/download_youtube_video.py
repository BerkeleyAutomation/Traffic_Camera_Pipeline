import sys, os

from tcp.configs.alberta_config import Config
from tcp.streaming.youtube_stream_downloader import YoutubeStreamDownloader

youtube_stream_url = 'https://www.youtube.com/watch?v=w54FXQjr0wQ'
stream_save_path = '/nfs/diskstation/jren/tcp_alberta_cam_test'

cnfg = Config()
downloader = YoutubeStreamDownloader(youtube_stream_url, stream_save_path, cnfg)

downloader.save_video('alberta_cam', save_frames=True)