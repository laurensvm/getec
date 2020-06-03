import logging
import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import numpy as np

from .genre import Genre


class Network(object):
    """
    An abstract base class which implements reshaping and other
    internal helper methods such as selecting the optimizer
    and setting up all the callback functions during training
    and saving and loading the models.

    """
    def __init__(self, model_directory, model=None, uid=None):
        self.model = model
        self.uid = uid
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
            self.build()

    def load_or_error(self):
        self.model = tf.keras.models.load_model(self.model_filepath)

    def train(self, train_ds, val_ds=None, epochs=1000):
        callbacks = self._get_callbacks(es_patience=200)

        train_ds = train_ds.map(self._reshape).batch(128)
        val_ds = val_ds.map(self._reshape).batch(128)

        return self.model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks)

    def evaluate(self, ds):
        ds = ds.map(self._reshape).batch(128)
        return self.model.evaluate(ds)

    def predict(self, X):
        return self.model.predict(X)

    def predict_visualize_layers(self, X):
        pass


class ConvNet(Network):
    """
    The implementation of the convolutional neural network.
    This class inherits from the abstract base class Network
    """
    def __init__(self, *args, **kwargs):
        super(ConvNet, self).__init__(*args, **kwargs)

    def _reshape(self, X, y):
        shape = tf.shape(X)
        res = tf.reshape(X, (shape[0], shape[1], 1))
        return res, y

    def build(self, input_shape=(11, 29, 1), optimizer='adam',
              conv1_units=32, conv2_units=64, dense_units=128):

        optimizer = self._select_optimizer(optimizer)

        model = tf.keras.Sequential([
            layers.Conv2D(conv1_units, (3, 3), padding="valid", activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(conv2_units, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(dense_units, activation='relu'),
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
        """
        Visualises the layers into matplotlib images
        :param X: Spectrogram data
        :return: Plot for various output layers.
        """

        if isinstance(X, np.ndarray):
            X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
        elif isinstance(X, tf.data.Dataset):
            X = X.map(self._reshape)

        for x, y in X:
            x = tf.expand_dims(x, 0)

            plt.title("Test Sample Input")
            plt.grid(False)
            plt.imshow(x[0, :, :, 0], aspect='auto', cmap='plasma', origin='lower')
            plt.colorbar()
            plt.show()

            layer_outputs = [layer.output for layer in self.model.layers]
            visualisation_model = tf.keras.models.Model(inputs=self.model.input, outputs=layer_outputs)

            visualisations = visualisation_model.predict(x)

            images_per_row = 4

            for layer_name, layer_activation in zip(map(lambda x : x.name, layer_outputs[:3]), visualisations[:3]):
                n_features = layer_activation.shape[-1]
                size = layer_activation.shape[1:3]
                n_cols = n_features // images_per_row
                grid = np.zeros((size[0] * n_cols, images_per_row * size[1]))

                for col in range(n_cols):
                    for row in range(images_per_row):
                        channel_image = layer_activation[0, :, :, col * images_per_row + row]
                        channel_image -= channel_image.mean()
                        channel_image /= channel_image.std()
                        channel_image *= 64
                        channel_image += 128
                        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                        grid[col * size[0]: (col + 1) * size[0], row * size[1]: (row + 1) * size[1]] = channel_image

                plt.figure(figsize=(1. / size[0] * grid.shape[1], 3. / size[1] * grid.shape[0]))
                plt.title(layer_name)
                plt.grid(False)
                plt.imshow(grid, aspect='auto', cmap='plasma', origin='lower')
                plt.colorbar()
                plt.show()

            pred = np.argmax(visualisations[-1])
            print(f"Predicted class: {Genre(pred)} with probability {visualisations[-1][0][pred]}\n"
                + f"Actual class: {Genre(y)}")


class RecurrentNet(Network):
    """
    The implementation of the recurrent neural network.
    This class inherits from the abstract base class Network
    """
    def __init__(self, *args, **kwargs):
        super(RecurrentNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam'):

        optimizer = self._select_optimizer(optimizer)

        model = tf.keras.Sequential([
            layers.SimpleRNN(256, input_shape=input_shape, return_sequences=True, activation='relu'),
            layers.SimpleRNN(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(Genre.count(), activation='softmax')
        ])

        model.build()
        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
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


class StandardNeuralNet(Network):
    """
    The implementation of the standard neural network.
    This class inherits from the abstract base class Network
    """

    def __init__(self, *args, **kwargs):
        super(StandardNeuralNet, self).__init__(*args, **kwargs)

    def build(self, input_shape=(11, 29),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='sgd'):

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


class ConvocurrentNet(Network):
    """
    The implementation of the combined convolutional and recurrent neural network.
    This class inherits from the abstract base class Network
    """
    def __init__(self, *args, **kwargs):
        super(ConvocurrentNet, self).__init__(*args, **kwargs)

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
        # extract = layers.Lambda(lambda x: x[:, :, 0: int(input_shape[1] / 2)])(_input)

        simple_rnn1 = layers.SimpleRNN(256, return_sequences=True)(_input)
        simple_rnn2 = layers.SimpleRNN(128)(simple_rnn1)
        dense2 = layers.Dense(128, activation="relu")(simple_rnn2)

        # Concatenate the two subnetworks
        concat = layers.Concatenate()([dense1, dense2])
        dense3 = layers.Dense(128, activation="relu")(concat)
        dropout1 = layers.Dropout(0.3)(dense3)
        dense4 = layers.Dense(128, activation="relu")(dropout1)
        output = layers.Dense(Genre.count(), activation='softmax')(dense4)

        model = tf.keras.Model(inputs=[_input], outputs=[output])

        model.summary()

        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        self.model = model
