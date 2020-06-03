import youtube_dl
import logging
import time

from youtube_dl.utils import DownloadError, ExtractorError

from os import path
from .genre import Genre


def download_playlists(songs_path, options=None):
    """
    Downloads all the playlists from youtube
    :param songs_path: The root directory for where the songs should be stored
    :param options: An optional object to configure downloading options
    :return: Saved files to the filesystem
    """

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
    playlist_genres["BALLAD"].append("https://www.youtube.com/playlist?list=PL-rFYrnzIalWsY6gkT7VDoYqpk8GL21cp")
    playlist_genres["ROCK"].append("https://www.youtube.com/playlist?list=PL6Lt9p1lIRZ311J9ZHuzkR5A3xesae2pk")
    playlist_genres["CLASSICAL"].append("https://www.youtube.com/playlist?list=PLxvodScTx2RtAOoajGSu6ad4p8P8uXKQk")
    playlist_genres["CLASSICAL"].append("https://www.youtube.com/playlist?list=PL2788304DC59DBEB4")
    playlist_genres["CLASSICAL"].append("https://www.youtube.com/playlist?list=PL68AC80CBF3649BBB")
    playlist_genres["HOUSE"].append("https://www.youtube.com/playlist?list=PLhInz4M-OzRUsuBj8wF6383E7zm2dJfqZ")
    playlist_genres["TECHNO"]\
        .append("https://www.youtube.com/playlist?list=PLriDNoSeceaRvxOGeEDItXnNoJ-H1l9g8")

    playlist_genres["HIPHOP"].append("https://www.youtube.com/playlist?list=PLvuMfxvpAQrkzez9insKS8cGPU74sK1Ss")
    # playlist_genres["HIPHOP"].append("https://www.youtube.com/playlist?list=PLetgZKHHaF-Zq1Abh-ZGC4liPd_CV3Uo4")
    playlist_genres["JAZZ"].append("https://www.youtube.com/playlist?list=PL8F6B0753B2CCA128")

    for genre, playlists in playlist_genres.items():

        logging.info("Downloading playlists from genre {0}".format(genre))
        options['outtmpl'] = path.join(songs_path, genre.lower(), '%(title)s.%(ext)s')

        with youtube_dl.YoutubeDL(options) as ydl:
            # Remove the cache from previous downloads as errors might arise otherwise
            ydl.cache.remove()
            try:
                ydl.download(playlists)
            except (DownloadError, ExtractorError) as e:
                logging.error(e)


def download_song(_path, url, options=None):
    """
    Download single song
    :param _path: filepath to store the single song
    :param url: YouTube/SoundCloud URL to retrieve the data from
    :param options: An optional object to configure downloading options
    :return: Saves the audio file to the filesystem
    """
    t = time.localtime()
    outtmpl = time.strftime("%H-%M-%S", t)

    if not options:
        options = {
            'format': 'bestaudio/best',
            'ignoreerrors': True,
            'cachedir': False,
            'outtmpl': path.join(_path, '{0}.%(ext)s'.format(outtmpl)),
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

    with youtube_dl.YoutubeDL(options) as ydl:
        # Remove the cache from previous downloads as errors might arise otherwise
        ydl.cache.remove()
        try:
            ydl.download([url])
        except (DownloadError, ExtractorError) as e:
            logging.error(e)

    return path.join(_path, outtmpl + ".mp3")
