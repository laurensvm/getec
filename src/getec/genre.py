from enum import Enum

class Genre(Enum):
    """
    This enum represents the different genres in integers, so that they can be
    stored alongside the spectrogram matrix
    """
    JAZZ = 0
    CLASSICAL = 1
    ROCK = 2
    HOUSE = 3
    TECHNO = 4
    HIPHOP = 5

    @staticmethod
    def count():
        return len([g for g in Genre])

    @staticmethod
    def from_directory_name(dirname):
        return Genre[dirname.upper()]

