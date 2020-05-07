import logging
import os

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from .genre import Genre


class Network(object):
    def __init__(self, model_directory, model=None, uid=None):
        self.model = model
        if uid:
            self.model_filepath = os.path.join(model_directory, self.__class__.__name__ + uid)
        else:
            self.model_filepath = os.path.join(model_directory, self.__class__.__name__)

    def _reshape(self, X, y):
        shape = tf.shape(X)
        res = tf.reshape(X, (shape[0], shape[1]))
        return res, y

    def _select_optimizer(self, optimizer):
        if optimizer == 'adam':
            return tf.keras.optimizers.Adam(lr=0.001)
        elif optimizer == 'sgd':
            return tf.keras.optimizers.SGD(lr=0.01)
        elif optimizer == 'RMSProp':
            return tf.keras.optimizers.RMSprop(lr=0.001)

    def _get_callbacks(self, es_patience=40, es_min_delta=0.001, es_monitor='loss',
                       cp_monitor='val_loss', lr_monitor='loss',
                       lr_patience=5, lr_factor=0.2, min_lr=0.0001):
        early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=es_monitor, mode='min', min_delta=es_min_delta, patience=es_patience)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                self.model_filepath, save_best_only=True,
                monitor=cp_monitor, mode='min', restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=lr_monitor, factor=lr_factor, patience=lr_patience, min_lr=min_lr)

        return [early_stopping, checkpoint, reduce_lr]

    def save(self):
        self.model.save(self.model_filepath)

    def build(self, optimizer=None):
        raise NotImplementedError()

    def load(self):
        try:
            self.model = tf.keras.models.load_model(self.model_filepath)
        except OSError:
            logging.error("Trying to load a model which does not exist. " +
                          "Creating the model instead")

    def train(self, train_ds, val_ds=None, epochs=1000):
        callbacks = self._get_callbacks()

        train_ds = train_ds.map(self._reshape).batch(128)
        val_ds = val_ds.map(self._reshape).batch(128)

        return self.model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_visualize_layers(self, X):
        pass


class ConvNet(Network):
    def __init__(self, *args, **kwargs):
        super(ConvNet, self).__init__(*args, **kwargs)

    def _reshape(self, X, y):
        shape = tf.shape(X)
        res = tf.reshape(X, (shape[0], shape[1], 1))
        return res, y

    def build(self, input_shape=(11, 29, 1), optimizer='adam'):

        optimizer = self._select_optimizer(optimizer)

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

    def predict_visualize_layers(self, X):

        if isinstance(X, np.ndarray):
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        elif isinstance(X, tf.data.Dataset):
            X = X.map(self._reshape).batch(128)

        layer_outputs = [layer.output for layer in self.model.layers]
        visualisation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)

        visualisations = visualisation_model.predict(X)

        print(visualisations)


class RecurrentNet(Network):
    def __init__(self, *args, **kwargs):
        super(RecurrentNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam'):

        optimizer = self._select_optimizer(optimizer)

        model = tf.keras.Sequential([
            layers.SimpleRNN(256, input_shape=input_shape, return_sequences=True),
            layers.SimpleRNN(128),
            layers.Dense(256, activation="relu"),
            # layers.SimpleRNN(256),
            # layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(Genre.count(), activation='softmax')
        ])

        # model = tf.keras.Sequential([
        #     layers.SimpleRNN(256, input_shape=input_shape, return_sequences=True),
        #     layers.Dropout(0.4),
        #     layers.SimpleRNN(128, return_sequences=True),
        #     layers.SimpleRNN(64),
        #     layers.Dense(256, activation="relu"),
        #     # layers.SimpleRNN(256),
        #     # layers.Dense(128, activation="relu"),
        #     layers.Dropout(0.2),
        #     layers.Dense(Genre.count(), activation='softmax')
        # ])

        model.build()
        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model

    def train(self, train_ds, val_ds=None, epochs=1000):
        callbacks = self._get_callbacks(es_patience=75)

        train_ds = train_ds.map(self._reshape).batch(128)
        val_ds = val_ds.map(self._reshape).batch(128)

        return self.model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks)


class StandardNeuralNet(Network):
    def __init__(self, *args, **kwargs):
        super(StandardNeuralNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam'):

        optimizer = self._select_optimizer(optimizer)

        model = tf.keras.Sequential([
            layers.Flatten(input_shape=input_shape),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dense(Genre.count(), activation='softmax')

        ])

        model.build()
        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model


class ConvucurrentNet(Network):
    def __init__(self, *args, **kwargs):
        super(ConvucurrentNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam'):

        # Input layer
        _input = layers.Input(shape=input_shape)

        # The convolutional part of the network

        # Reshape data to 3 dimensions
        reshape = layers.Reshape((input_shape[0], input_shape[1], 1))(_input)
        conv1 = layers.Conv2D(32, (3, 3), padding="valid", activation='relu')(reshape)
        max_pl1 = layers.MaxPooling2D((2, 2))(conv1)
        conv2 = layers.Conv2D(64, (3, 3), activation='relu')(max_pl1)
        flat = layers.Flatten()(conv2)

        dense1 = layers.Dense(128, activation='relu')(flat)

        # The recurrent part of the network. We need to slice the input and extract fewer samples
        extract = layers.Lambda(lambda x: x[:, :, 0: int(input_shape[1] / 2)])(_input)

        simple_rnn1 = layers.SimpleRNN(256, return_sequences=True)(extract)
        simple_rnn2 = layers.SimpleRNN(128)(simple_rnn1)
        dense2 = layers.Dense(256, activation="relu")(simple_rnn2)

        # Concatenate the two subnetworks
        concat = layers.Concatenate()([dense1, dense2])
        dropout1 = layers.Dropout(0.3)(concat)
        output = layers.Dense(Genre.count(), activation='softmax')(dropout1)

        model = tf.keras.Model(inputs=[_input], outputs=[output])

        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model
