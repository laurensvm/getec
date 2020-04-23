import logging

from file_handler import IOHandler
from preprocessor import PreProcessor

#temp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from matplotlib import cm


def spectrogram(samples, sample_rate, stride_ms=10.0,
                window_ms=20.0, max_freq=None, eps=1e-14):
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

if __name__ == "__main__":
    # configure_logging()
    handler = IOHandler()
    rate, data = handler.read_wav_file("Bellaire - Brasil.mp3")


    print(np.max(data))
    print(data.dtype)

    # n = len(data)
    # T = 1 / rate
    # yf = scipy.fft(data)
    # xf = np.linspace(0, int(rate / 2), int(n / 2))
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(xf, 2.0 / n * np.abs(yf[:int(n/2)]))
    # print(np.amax(2.0 / n * np.abs(yf[:int(n/2)])))
    # ax.ticklabel_format(useOffset=False, style='plain')
    # plt.grid()
    # plt.show()

    # spec = spectrogram(data, rate)
    # x = np.linspace(0, len(data) / rate, len(data))
    # plt.plot(x, data)
    # plt.show()
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(spec,aspect='auto', cmap=cm.get_cmap('PuBu'))
    # fig.colorbar(cax)
    # plt.show()
    nwindow = int(rate / 3)
    freqs, times, spec = signal.spectrogram(
        data,
        rate,
        window=signal.get_window('hann', nwindow),
        nperseg=nwindow,
        noverlap=nwindow/2,
        scaling='spectrum')

    spec_samp = spec[:, 0:10]
    test_samp = block_reduce(spec[:, 0:10], (4, 1), np.max)
    shape = test_samp.shape[0]
    plt.pcolormesh(np.linspace(0, 10, 10), np.linspace(0, 24000, shape), 10 * np.log10(test_samp))
    plt.show()
    # plt.show()
    # data = data.astype(np.int16)
    print(data.strides[0])
    # n = len(data)
    # T = 1 / rate
    # yf = scipy.fft(data)
    # xf = np.linspace(0, int(rate / 2), int(n / 2))
    #
    # fig, ax = plt.subplots()
    # ax.plot(xf, 2.0 / n * np.abs(yf[:int(n/2)]))
    # ax.ticklabel_format(useOffset=False, style='plain')
    # plt.grid()
    # plt.show()


