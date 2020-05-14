import logging
import numpy as np

from scipy import signal
from skimage.measure import block_reduce
from random import randint
import matplotlib.pyplot as plt
import tensorflow as tf

# from .formatter import to_tf_record_X_y
from .exceptions import (DownSampleException,
                         NotEnoughSamplesException,
                         RateException,
                         DataTypeException
                         )

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
    def get_audio_fragments(data, rate=RATE, n=10, sec=3):
        total_length = len(data)
        samples = []

        # Check if the sample has at least 120 seconds of sound
        if total_length < 2 * 60 * rate:
            raise NotEnoughSamplesException()

        # Strip the first and last 30 seconds
        begin_idx = 30 * rate
        end_idx = total_length - 30 * rate

        for _ in range(n):
            rand_idx = randint(begin_idx, end_idx - int(sec * rate))

            sample = data[rand_idx: rand_idx + int(sec * rate)]
            samples.append(sample)

        return samples

    @staticmethod
    def spectrogram(samples, sample_rate, stride_ms=10.0,
                    window_ms=20.0, eps=1e-14):
        """
        Replaced by scipy.spectrogram
        :param samples:
        :param sample_rate:
        :param stride_ms:
        :param window_ms:
        :param eps:
        :return:
        """
        samples = samples.astype(np.int16)

        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)

        # Extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(samples,
                                                  shape=nshape, strides=nstrides)

        assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

        # Window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]

        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft ** 2

        scale = np.sum(weighting ** 2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale

        # Prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

        # Compute spectrogram feature
        # ind = np.where(freqs <= max_freq)[0][-1] + 1
        specgram = np.log(fft + eps)
        return specgram

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
    def average_pool_spectrogram(spectrogram, block_size=(401, 1)):
        average_pooled_samples = block_reduce(spectrogram, block_size, np.mean)
        return average_pooled_samples

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
    def imshow(matrix):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.set_title("colorMap")
        plt.imshow(matrix)
        ax.set_aspect("equal")
        plt.colorbar(orientation="vertical")

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
    def preprocess_audio_data(rate, data, genre):
        if not isinstance(data, np.ndarray):
            raise DataTypeException()

        if rate != PreProcessor.RATE:
            raise RateException()

        samples = PreProcessor.get_audio_fragments(data, n=20, sec=3)
        processed = []

        for sample in samples:

            # First get the spectrogram representation. Then use max pooling to reduce the dimensionality
            freq, time, spectrogram = PreProcessor.get_spectrogram_data(sample, nwindow=int(0.2 * rate))
            spectrogram = PreProcessor.average_pool_spectrogram(spectrogram)

            spectrogram = PreProcessor.normalize_spectrogram(spectrogram)

            if not np.any(np.isnan(spectrogram)):
                if genre:
                    processed.append((spectrogram, genre.value))
                else:
                    processed.append(spectrogram)

        return processed

    @staticmethod
    def reshape_for_convnet(X):
        shape = tf.shape(X)
        return tf.reshape(X, [shape[0], shape[1], shape[2], 1])

    @staticmethod
    def reshape_for_rnn(data):
        return data[0], data[1]
