from pytube import Playlist

def url_grabber(playlistURL):
    p = Playlist(playlistURL)
    return p.video_urls
