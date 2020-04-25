import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

from .genre import Genre


def build_recurrent_network(input_shape=(307, 719)):
    model = tf.keras.Sequential()

    model.add(layers.GRU(256, input_shape=input_shape, return_sequences=True))
    model.add(layers.SimpleRNN(128))

    model.add(layers.Dense(Genre.count()))

    model.build()
    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer='sgd',
                  metrics=['accuracy'])

    return model

