import tensorflow as tf
import numpy as np

def to_tf_record_X_y(writer, X, y):

    def _int64_feature(value):
        if isinstance(value, np.ndarray):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
        else:
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _float_feature_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    feature = tf.train.Features(feature={
        'X': _float_feature_list(X.flatten()),
        'y': _int64_feature(y),
    })

    ex = tf.train.Example(features=tf.train.Features(feature=feature))

    writer.write(ex.SerializeToString())

def to_tf_record_X(writer, X):

    def _float_feature_list(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize_array(arr):
        return tf.io.serialize_tensor(arr)

    features = tf.train.Features(feature={
        'X': _bytes_feature(serialize_array(X)),
    })

    ex = tf.train.Example(features=features)

    writer.write(ex.SerializeToString())

feature_description = {
    'X': tf.io.FixedLenFeature([], tf.string)
}

def _parse_function(ex):
    example = tf.io.parse_single_example(ex, feature_description)
    feature = tf.io.parse_tensor(example["X"], out_type=tf.float64)
    return feature

def get_processed_matrix(ex):
    pass