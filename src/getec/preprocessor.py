import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

from .exceptions import DownSampleException, NotEnoughSamplesException

class PreProcessor(object):
    """
    This class handles all the processing of the
    audio data before we can feed it to the neural network
    """

    RATE = 44100

    def __init__(self):
        logging.debug("Initialised PreProcessor")

    @staticmethod
    def downsample(data, rate, new_rate=RATE):
        if rate < new_rate:
            raise DownSampleException()

        n_downsampled = int(new_rate / rate * len(data))
        return signal.resample(data, n_downsampled).astype(np.int16)

    @staticmethod
    def get_audio_fragment(self, data, rate=RATE, n=120):
        """
        This method extracts n seconds of audio in the middle of the audio
        :param data:
        :param rate:
        :param n:
        :return:
        """

        middle_idx = int(len(data) / 2)
        sample_range = n / 2 * rate

        if middle_idx - sample_range < 0:
            raise NotEnoughSamplesException()

        sample_fragment = data[middle_idx - sample_range: middle_idx + sample_range]

        logging.DEBUG("Extracted audio fragment from data which will be further processed.")

        return sample_fragment

    @staticmethod
    def get_spectrogram_data(self, data, rate, nwindow=None):
        """
        Gets the spectrogram representation of the data
        :param data:
        :param rate:
        :param nwindow:
        :return:
        """
        if not nwindow:
            nwindow = int(rate / 3)

        freqs, times, spec = signal.spectrogram(
            data,
            rate,
            window=signal.get_window('hann', nwindow),
            nperseg=nwindow,
            noverlap=nwindow / 2,
            scaling='spectrum')

        logging.DEBUG("Calculated spectrogram data.")
        return freqs, times, spec

    @staticmethod
    def max_pool_frequencies(self, data, block_size=(1, 1)):
        pass