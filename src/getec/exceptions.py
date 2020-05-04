"""
This file contains all the exceptions
"""

import logging

"""
Preprocessor Exceptions
"""


class DownSampleException(Exception):
    def __init__(self):
        logging.error("Cannot downsample signal as the required sample rate is too low")


class NotEnoughSamplesException(Exception):
    def __init__(self):
        logging.error("Failed extracting audio fragment. The audio data does not have enough samples")


class FileExtensionException(Exception):
    def __init__(self):
        logging.error("File extension does not match given extensions while changing extensions")


class PathDoesNotExistException(Exception):
    def __init__(self):
        logging.error("Path does not exist while reading wave file")


class DataTypeException(Exception):
    def __init__(self):
        logging.error("The datatype of the audio date given to the preprocessor is not of type np.array")


class InvalidBitSizeException(Exception):
    def __init__(self):
        logging.error("The wave file could not be read as it might have an invalid bit size")


class RateException(Exception):
    def __init__(self):
        logging.error("The file's audio rate is not 44100Hz, and downsampling is disabled. Skipping file")


