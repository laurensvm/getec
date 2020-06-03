import tensorflow as tf
import numpy as np

class Dataset(object):

    """
    This class is responsible for all the communication with the
    .tfrecord dataset. It serializes and deserializes the data.
    """

    _feature_description = {
        'X': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    def __init__(self, processed_filepath, buffer_size=100000):
        self.dataset_filepaths = [processed_filepath]
        self.ds = tf.data.TFRecordDataset(self.dataset_filepaths)\
            .map(Dataset._parse_function)\
            .cache()\
            .shuffle(buffer_size, seed=42, reshuffle_each_iteration=True)

        self.ds_train, self.ds_val, self.ds_test = self._split()
        self.buffer_size = buffer_size

    @staticmethod
    def _parse_function(ex):
        """
        This method gets applied to every example to deserialize it
        :param ex: An example from the dataset
        :return: The spectrogram matrix and the genre
        """
        example = tf.io.parse_single_example(ex, Dataset._feature_description)
        X = tf.io.parse_tensor(example["X"], out_type=tf.float64)
        return X, example["y"]

    def _split(self, size=22700): # n = 22500
        """
        Splits the dataset into training, validation and testing sets.
        :param size: The approximate size of the dataset
        :return:
        """
        val_size = int(0.15 * size)
        test_size = int(0.15 * size)

        val_ds = self.ds.take(val_size)
        test_ds = self.ds.skip(val_size).take(test_size)
        train_ds = self.ds.skip(val_size).skip(test_size)

        return train_ds, val_ds, test_ds

    def to_tf_record_X_y(self, X, y):
        """
        Serialize spectrogram matrix and genre into bytes to store it in the .tfrecord file
        :param X: The spectrogram
        :param y: The genre label
        :return: serialized example
        """

        def _int64_feature(value):
            if isinstance(value, np.ndarray):
                return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            else:
                return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy()
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def serialize_array(arr):
            return tf.io.serialize_tensor(arr)

        features = tf.train.Features(feature={
            'X': _bytes_feature(serialize_array(X)),
            'y': _int64_feature(y),
            # 'name': _bytes_feature(str.encode(name))
        })

        ex = tf.train.Example(features=features)

        return ex.SerializeToString()

    def get_training_set(self):
        return self.ds_train

    def get_validation_set(self):
        return self.ds_val

    def get_test_set(self):
        return self.ds_test