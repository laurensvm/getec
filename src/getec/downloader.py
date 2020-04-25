import youtube_dl
import logging

from os import path
from .genre import Genre


def download_playlists(songs_path, options=None):

    if not options:
        options = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
        }

    playlist_genres = { g.name: [] for g in Genre }

    # Add playlist links
    playlist_genres["JAZZ"].append("https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128")

    for genre, playlists in playlist_genres.items():
        logging.info("Downloading playlists from genre {0}".format(genre))
        options['outtmpl'] = path.join(songs_path, genre.lower(), '%(title)s.%(ext)s')

        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download(playlists)
