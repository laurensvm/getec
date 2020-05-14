import argparse, logging

from getec import *

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix

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

    parser.add_argument(
        "--load_from_file", "-loadfile",
        action="store_const",
        const=True,
        help="Load preprocessed audio data from the saved file to perform direct training. " +
             "This will skip downloading songs and preprocessing them."
    )

    parser.add_argument(
        "--predict_from_youtube_video", "-predict_youtube_video",
        help="Enter a url, and the application will download the youtube song, analyse it, and predict the genre"
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

        self.io = IOHandler(songs_path=args.songs_path)

        if args.download_songs:
            self.download_data()

        self.dataset = Dataset(self.io.processed_filepath)

        if args.predict_from_youtube_video:
            self.predict_genre_for_youtube_video(args.predict_from_youtube_video)


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

        writer = tf.io.TFRecordWriter(self.io.processed_filepath)

        for genre in genres:
            song_paths = self.io.get_filepaths_for_genre(genre)

            for song in song_paths:
                try:
                    rate, data = self.io.read(os.path.join(song["path"], song["name"]))
                except InvalidBitSizeException:
                    continue

                try:
                    processed = PreProcessor.preprocess_audio_data(rate, data, genre)
                except (NotEnoughSamplesException, DataTypeException, RateException):
                    continue

                for (X, y) in processed:
                    writer.write(self.dataset.to_tf_record_X_y(X, y))

                # Clear cash if more than 10 wav files in cashed folder
                if self.io.check_cache() > 10:
                    self.io.clear_cache()

        writer.close()

    def train_model(self, model=ConvocurrentNet):

        _model = model(self.io.model_directory, uid="ballad")
        # _model.build(input_shape=(11, 24, 1))
        # _model.build(input_shape=(11, 14))
        # _model.build(input_shape=(11, 9))
        _model.build(input_shape=(11, 29))

        # In case there exists a model
        # _model.load()

        train_ds = self.dataset.get_training_set()
        val_ds = self.dataset.get_validation_set()

        hist = _model.train(train_ds, val_ds=val_ds, epochs=1000)

        Visualizer(self.io.get_image_directory()).plot_history(hist.history, _model)

        _model.save()

    def test_model(self, model=ConvocurrentNet, uid=None):
        _model = model(self.io.model_directory, uid=uid)
        _model.load()

        # Test model
        test_ds = self.dataset.get_test_set()

        stats = _model.evaluate(test_ds)

        return stats

    def download_data(self):
        logging.info("Downloading audio data from youtube playlists specified in downloader.py")
        download_playlists(self.io.songs_path)

    def preprocess_file(self, filename, genre=None):
        if self.io.ispath(filename):
            rate, data = self.io.read(filename)
        else:
            filepath = self.io.build_filepath(filename, genre)
            rate, data = self.io.read(filepath)

        try:
            return PreProcessor.preprocess_audio_data(rate, data, genre)
        except DataTypeException:
            return

    def predict_genre_for_file(self, filepath, network=ConvNet):
        processed = self.preprocess_file(filepath)

        model = network(self.io.model_directory)
        model.load()

        preds = model.predict(np.array(processed))

        preds = list(map(lambda x: np.argmax(x), preds))

        from collections import Counter
        c = Counter(preds)

        first = c.most_common(1)[0]
        # sec = c.most_common(2)[1]
        print("Predicted class is: {0} with {1} / 20 classifications".format(Genre(first[0]), first[1]))
        # print("Second predicted class: {0} with {1} / 20 classifications".format(Genre(sec[0]), sec[1]))

    def predict_genre_for_youtube_video(self, url):
        filepath = download_song(self.io.get_temp_filepath(), url)

        self.predict_genre_for_file(filepath)

    def test_models_performance(self):
        uids = ["sec1", "sec1_5", "sec2", "sec2_5", None]
        models = [RecurrentNet, ConvocurrentNet, ConvNet, StandardNeuralNet]

        stats = []
        for _model in models:
            _models = []
            for _uid in uids:
                model = _model(self.io.model_directory, uid=_uid)

                try:
                    model.load_or_error()
                except OSError:
                    continue

                if _uid:
                    test_ds = Dataset(self.io.get_processed_filepath(f"processed_{_uid}.tfrecord")).get_test_set()
                else:
                    test_ds = self.dataset.get_test_set()

                evaluated = model.evaluate(test_ds)
                _models.append((evaluated, model))
            stats.append(_models)

        Visualizer(self.io.get_image_directory()).plot_models_performances(stats)

    def visualise_layers(self):
        test_ds = self.dataset.get_test_set().take(1)

        model = ConvNet(self.io.model_directory)
        model.load_or_error()
        model.predict_visualize_layers(test_ds)



if __name__ == "__main__":
    args = parse_args()

    app = App(args)

    # app.download_data()
    # app.train_model()
    # app.perform_batch_preprocessing()
    # app.test_models_performance()
    # app.test_model()

    app.visualise_layers()

