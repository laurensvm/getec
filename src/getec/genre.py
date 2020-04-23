from enum import Enum

class Genre(Enum):
    """
    This enum represents the different genres in integers, so that they can be
    stored alongside the spectrogram matrix
    """
    JAZZ = 1
    CLASSICAL = 2
    ROCK = 3
    HOUSE = 4
    TECHNO = 5
    HIPHOP = 6

    def from_directory_name(self, dirname):
        return Genre[dirname.upper()]