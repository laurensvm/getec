import argparse, logging

from getec import *

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def parse_args():
    """
    We handle all the input args and configure the program using the args
    :return:
    """

    parser = argparse.ArgumentParser(
        prog="Genre Detector (gentec)",
        description="Run the genre detector pipeline with specified arguments",

    )
    parser.add_argument(
        "--debug", "-d",
        action="store_const",
        const=True,
        help="Run in debug mode."
    )

    parser.add_argument(
        "--minimal", "-m",
        action="store_const",
        const=True,
        help="Run in minimal mode. This means that the application will use minimal " +
             "overhead. No debug logs and minimal preprocessing."
    )

    parser.add_argument(
        "--songs_path", "-sp",
        help="Specify the absolute path of the songs folder. If not specified " +
             "the program will assume this to be a directory in the " +
             "parent directory of the source directory with the name 'songs'"

    )

    parser.add_argument(
        "--downsample", "-downsample",
        action="store_const",
        const=True,
        help="If selected, higher sampled audio than 44100Hz will be downsampled to 44100Hz. " +
             "\n Be aware that this yields higher preprocessing times"
    )

    parser.add_argument(
        "--download_songs", "-download",
        action="store_const",
        const=True,
        help="Download all the songs specified in playlists before performing preprocessing and " +
             "network training. On first run, this argument must be enabled to gather data"
    )

    return parser.parse_args()


class App(object):
    """
    This class manages all the processes of the application
    """

    def __init__(self, args):
        """
        Initialise the app using arguments
        :param args:
        """
        if args.minimal:
            self.debug = False
        else:
            self.debug = args.debug
            self.minimal = False

            self.configure_logging()

        self.downsample = args.downsample

        self.io_handler = IOHandler(songs_path=args.songs_path)

        if args.download_songs:
            self.download_data()


    def configure_logging(self):
        """
        Configures the logging engine
        :return:
        """
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler()
            ]
        )

        logging.info("Logger initialised")

    def perform_batch_preprocessing(self, genres=None):
        # If no list of genres is specified, perform preprocessing on all
        if not genres:
            genres = [g for g in Genre]

        processed_songs = []

        for genre in genres:
            song_paths = self.io_handler.get_filepaths_for_genre(genre)

            for song in song_paths:
                rate, data = self.io_handler.read_wav_file(song)

                try:
                    data = self.preprocess_audio_data(rate, data, genre)
                except (NotEnoughSamplesException, DataTypeException):
                    continue

                processed_songs.append(data)

                # Clear cash if more than 10 wav files in cashed folder
                if self.io_handler.check_cash() > 10:
                    self.io_handler.clear_cash()

        self.train_model(processed_songs)

    def preprocess_audio_data(self, rate, data, genre):
        if not isinstance(data, np.ndarray):
            raise DataTypeException()

        if self.downsample:
            data = PreProcessor.downsample(data, rate)

        samples = PreProcessor.get_audio_fragment(data)

        # First get the spectrogram representation. Then use max pooling to reduce the dimensionality
        freq, time, spectrogram = PreProcessor.get_spectrogram_data(samples)
        freq, spectrogram = PreProcessor.max_pool_spectrogram(spectrogram)

        spectrogram = PreProcessor.normalize_spectrogram(spectrogram)

        label_encoding = np.full((spectrogram.shape[0], 1), genre.value)

        # First column of the matrix is the encoded genre
        return np.append(label_encoding, spectrogram, axis=1)

    def train_model(self, processed_data):
        model = build_recurrent_network()

        genre_encodings = list(map(lambda x: x[:,0], processed_data))
        y_train = PreProcessor.encode_genre_to_vector(genre_encodings)
        x_train = list(map(lambda x: x[:, 1:], processed_data))

        input = np.array(x_train)
        model.fit(input, np.array(y_train), epochs=5)

        preds = model.predict(input)

        print(preds)
        print(np.argmax(preds), np.array(y_train))

    def download_data(self):
        logging.info("Downloading audio data from youtube playlists specified in downloader.py")
        download_playlists(self.io_handler.songs_path)

    def preprocess_file(self, filename, genre):
        filepath = self.io_handler.build_filepath(filename, genre)
        rate, data = self.io_handler.read_wav_file(filepath)

        try:
            return self.preprocess_audio_data(rate, data, genre)
        except DataTypeException:
            return

if __name__ == "__main__":
    args = parse_args()
    app = App(args)
    # app.download_data()
    app.perform_batch_preprocessing()
