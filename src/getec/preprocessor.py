import logging
import numpy as np

from scipy import signal
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

from .exceptions import DownSampleException, NotEnoughSamplesException
from .genre import Genre

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
    def get_audio_fragment(data, rate=RATE, n=120):
        """
        This method extracts n seconds of audio in the middle of the audio
        :param data:
        :param rate:
        :param n:
        :return:
        """

        middle_idx = int(len(data) / 2)
        sample_range = int(n / 2 * rate)

        if middle_idx - sample_range < 0:
            raise NotEnoughSamplesException()

        sample_fragment = data[middle_idx - sample_range: middle_idx + sample_range]

        logging.debug("Extracted audio fragment from data which will be further processed.")

        return sample_fragment

    @staticmethod
    def get_spectrogram_data(data, rate=RATE, nwindow=None):
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

        spec = spec.astype(np.int16)

        logging.debug("Calculated spectrogram data.")
        return freqs, times, spec

    @staticmethod
    def max_pool_spectrogram(spectrogram, block_size=(24, 1)):
        """
        This method reduces the frequency domain by max pooling
        :param spectrogram:
        :param block_size:
        :return:
        """
        max_pooled_samples = block_reduce(spectrogram, block_size, np.max)
        freq = np.linspace(0, PreProcessor.RATE / 2, max_pooled_samples.shape[0])
        return freq, max_pooled_samples

    @staticmethod
    def plot_spectrogram(spectrogram, time=None, freq=None):
        """
        Plots the spectrogram
        :param spectrogram: The spectrogram
        :param time: A 1d vector consisting of the linear timesteps
        :param freq: A 1d vector consisting of the linear frequency bins
        :return: Show plot
        """
        if type(time) == None:
            time = np.linspace(0, 120, spectrogram.shape[1])

        if type(freq) == None:
            freq = np.linspace(0, PreProcessor.RATE, spectrogram.shape[0])

        plt.pcolormesh(time, freq, 10 * np.log10(spectrogram))
        plt.show()

    @staticmethod
    def normalize_spectrogram(s):
        """
        Normalizes the spectrogram into values ranging from [0, 1]
        :param s: The spectrogram
        :return: The normalized spectrogram
        """
        s_max, s_min = s.max(), s.min()
        s = (s - s_min) / (s_max - s_min)
        return s

    @staticmethod
    def encode_genre_to_vector(genre):
        """
        This method takes a genre and returns a vector with zeros except
        for the index of the corresponding genre.
        :param genre:
        :return:
        """

        if type(genre) == list:
            return np.fromiter(map(lambda x: x[0], genre), dtype=np.int8)
        else:
            return genre.value

