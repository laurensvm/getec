import youtube_dl
import logging

from youtube_dl.utils import DownloadError, ExtractorError

from os import path
from .genre import Genre


def download_playlists(songs_path, options=None):

    if not options:
        options = {
            'format': 'bestaudio/best',
            'ignoreerrors': True,
            'cachedir': False,

            'download_archive': path.join(songs_path, "archive.txt"),
            'downloader': [{
               'continuedl': True
            }],
            'postprocessor_args': ["-ar", "44100"],
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
        }

    playlist_genres = { g.name: [] for g in Genre }

    # Add playlist links
    playlist_genres["ROCK"].append("https://www.youtube.com/playlist?list=PL6Lt9p1lIRZ311J9ZHuzkR5A3xesae2pk")
    playlist_genres["CLASSICAL"].append("https://www.youtube.com/playlist?list=PLxvodScTx2RtAOoajGSu6ad4p8P8uXKQk")
    playlist_genres["HOUSE"].append("https://www.youtube.com/playlist?list=PLhInz4M-OzRUsuBj8wF6383E7zm2dJfqZ")
    playlist_genres["TECHNO"]\
        .append("https://www.youtube.com/playlist?list=PLriDNoSeceaRvxOGeEDItXnNoJ-H1l9g8")

    playlist_genres["HIPHOP"].append("https://www.youtube.com/playlist?list=PLvuMfxvpAQrkzez9insKS8cGPU74sK1Ss")
    playlist_genres["HIPHOP"].append("https://www.youtube.com/playlist?list=PLetgZKHHaF-Zq1Abh-ZGC4liPd_CV3Uo4")
    playlist_genres["JAZZ"].append("https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128")

    for genre, playlists in playlist_genres.items():
        #TEMP
        if genre == "ROCK" or genre == "JAZZ" or genre == "CLASSICAL" or genre == "HOUSE":
            continue

        logging.info("Downloading playlists from genre {0}".format(genre))
        options['outtmpl'] = path.join(songs_path, genre.lower(), '%(title)s.%(ext)s')

        with youtube_dl.YoutubeDL(options) as ydl:
            # Remove the cache from previous downloads as errors might arise otherwise
            ydl.cache.remove()
            try:
                ydl.download(playlists)
            except (DownloadError, ExtractorError) as e:
                logging.error(e)
