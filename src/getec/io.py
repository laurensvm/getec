import os

from os import path
from pathlib import Path
from shutil import copyfile
from pydub import AudioSegment
from scipy.io import wavfile

import logging

from .exceptions import PathDoesNotExistException, FileExtensionException, InvalidBitSizeException


class IOHandler(object):
    """
    This class handles all the interaction with the file system.
    It converts .mp3 files to .wav files
    """

    SONGS_DIRECTORY = "songs"
    CACHED_DIRECTORY = "cached"
    PROCESSED_FILENAME = "processed.tfrecord"
    IMAGE_DIRECTORY = "figs"

    def __init__(self, songs_path=None, processed_filepath=None):

        # If no songs path is specified, look for default path in root directory
        if not songs_path:
            self.basedir = path.dirname(path.dirname(path.dirname(path.realpath(__file__))))
            songs_path = path.join(self.basedir, IOHandler.SONGS_DIRECTORY)

        self.songs_path = songs_path

        # Make a cashed folder inside the songs folder
        # This folder will contain all the .wave files
        # which are currently being processed
        if path.exists(self.songs_path):
            self.cashed_path = path.join(self.songs_path, IOHandler.CACHED_DIRECTORY)
            Path(path.join(self.cashed_path)).mkdir(parents=True, exist_ok=True)

        # Make a filepath for the processed.tfrecord dataset
        if not processed_filepath:
            self.processed_filepath = path.join(self.basedir, IOHandler.PROCESSED_FILENAME)
        else:
            self.processed_filepath = processed_filepath

        # Make a directory for the saved models
        self.model_directory = path.join(self.basedir, "saved_models")

    def mp3_to_wav(self, song_path):
        """
        Converts the song from .mp3 format into .wav format
        :param song:
        :return:
        """

        # Check if file is of type wave already
        # if so, return the filepath
        if song_path.endswith(".wav", 4):
            return song_path

        if not path.exists(song_path):
            raise PathDoesNotExistException()

        # Get the song name
        song_name = path.split(song_path)[1]

        # Copy the file to the cashed folder and change extension
        cashed_source_path = path.join(self.cashed_path, change_extension(song_name))

        # Check if the file already exists in the cashed source path
        # if so, return the cashed source path
        if path.exists(cashed_source_path):
            logging.debug("Song {0} does already exist in cashed folder".format(song_name))
            return cashed_source_path
        else:
            copyfile(song_path, cashed_source_path)

        # Converts the song from .mp3 to .wav
        logging.info("Converting song {0} from .mp3 to .wav".format(song_name))
        converter = AudioSegment.from_mp3(cashed_source_path)
        converter.export(cashed_source_path, format="wav")

        return cashed_source_path

    def read(self, song_path):
        """
        Reads a wave file from the file system and returns a numpy array with the
        audio data, as well as the sample rate
        :param song_path:
        :return:
        """

        # Check if the src path is valid
        if not path.exists(song_path):
            raise PathDoesNotExistException()

        logging.debug("Reading wave file from path: {0}".format(song_path))

        # If the file is of type mp3, we automatically convert to wave
        if song_path.endswith(".mp3"):
            song_path = self.mp3_to_wav(song_path)

        try:
            rate, audio_data = wavfile.read(song_path)
        except ValueError:
            raise InvalidBitSizeException()

        return rate, audio_data.T[0]

    def get_filepaths_for_genre(self, genre):
        """
        Get all the filepaths from songs in specified genre
        The folder serves as the label for the audio data
        :param genre:
        :return: A list of song paths from the specified genre
        """
        genre_path = path.join(self.songs_path, genre.name.lower())
        if path.exists(genre_path):
            song_paths = [s for s in os.listdir(genre_path) if path.isfile(path.join(genre_path, s))]
            return [{"name": s, "path": genre_path} for s in song_paths if not s.startswith(".")]
        return [{}]

    def clear_cash(self):
        """
        Clears all .wav files in the cash folder
        :return:
        """
        for file in os.listdir(self.cashed_path):
            if path.isfile(path.join(self.cashed_path, file)) and file.endswith(".wav"):
                os.remove(path.join(self.cashed_path, file))

    def check_cash(self):
        """
        Checks how many wave files are present in the cashed folder
        :return: Amount of wave files in cashed folder
        """
        return len([f for f in os.listdir(self.cashed_path) if path.isfile(path.join(self.cashed_path, f))])

    def get_model_filepath(self, name="generic_model"):
        return path.join(self.model_directory, name)

    def get_model_directory(self):
        return self.model_directory

    def ispath(self, filename):
        return path.exists(filename)

    def get_temp_filepath(self):
        return path.join(self.basedir, "tmp")

    def get_image_directory(self):
        return path.join(path.dirname(self.basedir), IOHandler.IMAGE_DIRECTORY)


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
        raise FileExtensionException()

    return song

