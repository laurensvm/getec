from os import path
from pathlib import Path
from shutil import copyfile
from pydub import AudioSegment
from scipy.io import wavfile

import logging


class IOHandler(object):
    """
    This class handles all the interaction with the file system.
    It converts .mp3 files to .wav files
    """

    SONGS_DIRECTORY = "songs"

    def __init__(self, songs_path=None):

        # If no songs path is specified, look for default path in root directory
        if not songs_path:
            basedir = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
            songs_path = path.join(basedir, IOHandler.SONGS_DIRECTORY)

        self.songs_path = songs_path

        # Make a cashed folder inside the songs folder
        # This folder will contain all the .wave files
        # which are currently being processed
        if path.exists(self.songs_path):
            self.cashed_path = path.join(self.songs_path, "cashed")
            Path(path.join(self.cashed_path)).mkdir(parents=True, exist_ok=True)


    def get_source_path(self, song):
        """
        Returns the full path of the song given that the song
        resides in the songs path
        :param song:
        :return:
        """
        return path.join(self.songs_path, song)



    def mp3_to_wav(self, song):
        """
        Converts the song from .mp3 format into .wav format
        :param song:
        :return:
        """

        # Check if file is of type wave already
        # if so, return the filepath
        if song.endswith(".wav", 4):
            return song

        source_path = path.join(self.songs_path, song)
        if not path.exists(source_path):
            raise IOError("Specified song does not exist in path: {0}".format(self.songs_path))

        # Copy the file to the cashed folder and change extension
        cashed_source_path = path.join(self.cashed_path, change_extension(song))

        # Check if the file already exists in the cashed source path
        # if so, return the cashed source path
        if path.exists(cashed_source_path):
            logging.debug("Song {0} does already exist in cashed folder".format(song))
            return cashed_source_path
        else:
            copyfile(source_path, cashed_source_path)

        # Converts the song from .mp3 to .wav
        logging.info("Converting song {0} from .mp3 to .wav".format(song))
        converter = AudioSegment.from_mp3(cashed_source_path)
        converter.export(cashed_source_path, format="wav")

        return cashed_source_path

    def read_wav_file(self, song):
        """
        Reads a wave file from the file system and returns a numpy array with the
        audio data, as well as the sample rate
        :param src:
        :return:
        """

        logging.debug("Reading wave file {0}".format(song))

        src = path.join(self.songs_path, song)

        # Check if the src path is valid
        if not path.exists(src):
            raise IOError("Path does not exist while reading wave file")

        # If the file is of type mp3, we automatically convert to wave
        if song.endswith(".mp3"):
            src = self.mp3_to_wav(song)

        rate, audio_data = wavfile.read(src)

        return rate, audio_data.T[0]


def change_extension(song, from_ext=".mp3", to_ext=".wav"):
    """
    Changes the file name string representation from extension to extension
    :param song:
    :param from_ext:
    :param to_ext:
    :return:
    """
    if song.endswith(to_ext):
        return song

    if song.endswith(from_ext):
        song = song[:-len(from_ext)]
        song += to_ext
    else:
        raise IOError("File extension does not match given extensions while changing extensions")

    return song

