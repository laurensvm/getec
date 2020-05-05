import logging
import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from .genre import Genre

class Network(object):
    def __init__(self, model_directory, model=None):
        self.model = model
        self.model_filepath = os.path.join(model_directory, self.__class__.__name__)

    def _get_callbacks(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss', mode='min', min_delta=0.001, patience=40)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                self.model_filepath, save_best_only=True,
                monitor='val_loss', mode='min', restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.2, patience=5, min_lr=0.0001)

        return [early_stopping, checkpoint, reduce_lr]

    def save(self):
        self.model.save(self.model_filepath)

    def build(self):
        pass

    def load(self):
        try:
            self.model = tf.keras.models.load_model(self.model_filepath)
        except OSError:
            logging.error("Trying to load a model which does not exist. " +
                          "Creating the model instead")

    def train(self, X, y, validation_data=None, epochs=1000):
        callbacks = self._get_callbacks()
        return self.model.fit(X, y, validation_data=validation_data, epochs=epochs, callbacks=callbacks)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ConvNet(Network):
    def __init__(self, *args, **kwargs):
        super(ConvNet, self).__init__(*args, **kwargs)

    def _reshape(self, X, y):
        shape = tf.shape(X)
        res = tf.reshape(X, (shape[0], shape[1], 1))
        return (res, y)

    def build(self, input_shape=(11, 29, 1), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)):
        model = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), padding="valid", activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),

            # layers.Dense(256, activation='relu'),
            # layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(Genre.count(), activation='softmax')
        ])

        model.build()
        model.summary()

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        self.model = model

    def train(self, train_ds, val_ds=None, epochs=1000):
        callbacks = self._get_callbacks()

        train_ds = train_ds.map(self._reshape).batch(128)
        val_ds = val_ds.map(self._reshape).batch(128)

        return self.model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks)

    def evaluate(self, test_ds):
        test_ds = test_ds.map(self._reshape).batch(128)

        return self.model.evaluate(test_ds)

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        return self.model.predict(X)


class RecurrentNet(Network):
    def __init__(self, *args, **kwargs):
        super(RecurrentNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.SGD(lr=0.00001)):

        model = tf.keras.Sequential([
            layers.GRU(128, recurrent_dropout=0.1, input_shape=input_shape, return_sequences=True),
            layers.Dense(128, activation="relu"),
            layers.SimpleRNN(64, dropout=0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(Genre.count(), activation='softmax')
        ])

        model.build()
        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model



