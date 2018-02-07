import youtube_dl

from tcp.streaming.stream_downloader import StreamDownloader

class M3U8Logger(object):

    def debug(self, msg):
        self.url = msg

    def warning(self, msg):
        print('[WARNING]' + msg)

    def error(self, msg):
        print('[ERROR]' + msg)

"""
    Format parameter guide:
    format code  extension  resolution note
    92           mp4        240p       HLS , h264, aac  @ 48k
    93           mp4        360p       HLS , h264, aac  @128k
    94           mp4        480p       HLS , h264, aac  @128k
    95           mp4        720p       HLS , h264, aac  @256k
    96           mp4        1080p      HLS , h264, aac  @256k (best)
"""
def get_youtube_m3u8_url(youtube_url):
    m3u8 = M3U8Logger()
    ydl_opts = {
        'format': '95',
        'simulate': True,
        'forceurl': True,
        'quiet': True,
        'logger': m3u8,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
        return m3u8.url

class YoutubeStreamDownloader(StreamDownloader):

    def __init__(self, youtube_url, stream_save_path, config):
        self.youtube_url = youtube_url
        stream_src = get_youtube_m3u8_url(self.youtube_url)
        super(YoutubeStreamDownloader, self).__init__(stream_src, stream_save_path, config)

    def save_video(self, filename, save_frames=False):
        dir_size_maxed = False
        while not dir_size_maxed:
            self.stream_src = get_youtube_m3u8_url(self.youtube_url)
            dir_size_maxed = super(YoutubeStreamDownloader, self).save_video(filename, save_frames)
