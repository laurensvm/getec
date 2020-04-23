"""
This file contains all the exceptions
"""

import logging

"""
Preprocessor Exceptions
"""
class DownSampleException():
    def __init__(self):
        logging.ERROR("Cannot downsample signal as the required sample rate is too low")

class NotEnoughSamplesException():
    def __init__(self):
        logging.ERROR("Failed extracting audio fragment. The audio data does not have enough samples")

