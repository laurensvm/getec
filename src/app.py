import argparse, logging

from getec import IOHandler, PreProcessor

import numpy as np
import matplotlib.pyplot as plt

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

    def main(self):

        # TEMP
        rate, data = self.io_handler.read_wav_file("Dave Brubeck - Take Five.mp3")

        if self.downsample:
            data = PreProcessor.downsample(data, rate)


        samples = PreProcessor.get_audio_fragment(data)

        # First get the spectrogram representation. Then use max pooling to reduce the dimensionality
        freq, time, spectrogram = PreProcessor.get_spectrogram_data(samples)
        freq, spectrogram = PreProcessor.max_pool_spectrogram(spectrogram)

        spectrogram = PreProcessor.normalize_spectrogram(spectrogram)

        PreProcessor.plot_spectrogram(spectrogram, time, freq)


if __name__ == "__main__":
    args = parse_args()
    app = App(args)
    app.main()

