from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import TimeDistributed


class MyModel(metaclass=ABCMeta):
    def __init__(self, num_classes, shape=None, nbout=None, momentum=0):
        self.num_classes = num_classes
        self.shape = shape
        self.nbout = nbout
        self.momentum = momentum

        self.training_history = None
        self.evaluation_accuracy = None
        self.evaluation_loss = None
        self.predictions = None
        self.predictions_single = None

        self.model = self.define_model()
        self.predictor = self.define_predictor()

    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def validate_dataset(self):
        # compare my expected shape vs. dataset
        # True: will work
        # False: recommendation?
        pass


    def define_predictor(self):
        # probability predictor
        return tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])

    def compile(self):
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=["accuracy"],
        )

    def run(self, train_ds, val_ds, epochs):
        self.compile()
        self.train(train_ds, val_ds, epochs)
        self.evaluate(val_ds)

    def train(self, train_ds, val_ds, epochs):
        history = self.model.fit(
            train_ds, validation_data=val_ds, epochs=epochs
        )
        self.training_history = history.history

    def evaluate(self, ds):
        self.evaluation_loss, self.evaluation_accuracy = self.model.evaluate(ds)

    def predict(self, ds):
        self.predictions = self.predictor.predict(ds)
        self.predictions_single = np.argmax(self.predictions, axis=1)


class ModelType1(MyModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def define_model(self):
        """Construct your AI layers.

        Pick optimizer, loss function & accuracy metrics are arts. This is
        the core decision of how your work might be different from others.

        """
        # from another tutorial
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.num_classes),
            ]
        )
        return model


class ModelCRNN(MyModel):
    def __init__(
        self, num_classes, shape=(20, 112, 112, 3), nbout=2, momentum=0.9, dataset=None
    ):
        super().__init__(num_classes, shape, nbout, momentum)


    def define_model(self):
        # CNN part

        model = tf.keras.Sequential()

        # CNN part here
        model.add(
            TimeDistributed(
                tf.keras.layers.experimental.preprocessing.RandomRotation(
                    (-0.3, 0.3)
                )
            )
        )

        model.add(
            TimeDistributed(
                Conv2D(
                    32,
                    (3, 3),
                    input_shape=self.shape,
                    padding="same",
                    activation="relu",
                ),
                input_shape=self.shape,
            )
        )
        model.add(TimeDistributed(MaxPool2D(strides=(2, 2))))

        model.add(
            TimeDistributed(
                Conv2D(
                    32,
                    (3, 3),
                    input_shape=self.shape,
                    padding="same",
                    activation="relu",
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization(momentum=self.momentum)))
        model.add(TimeDistributed(MaxPool2D(strides=(2, 2))))

        model.add(
            TimeDistributed(
                Conv2D(
                    32,
                    (3, 3),
                    input_shape=self.shape,
                    padding="same",
                    activation="relu",
                )
            )
        )
        model.add(TimeDistributed(BatchNormalization(momentum=self.momentum)))
        model.add(TimeDistributed(MaxPool2D(strides=(2, 2))))

        model.add(TimeDistributed(GlobalMaxPool2D()))

        # here, you can also use GRU or LSTM, RNN
        model.add(GRU(256, activation="relu", return_sequences=True))
        model.add(GRU(256, activation="relu"))

        # and finally, we make a decision network, Dense layer
        model.add(Dense(1024, activation="relu"))

        model.add(Dropout(0.5))

        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))

        # output layers
        model.add(Dense(self.nbout, activation="softmax"))

        return model


class ModelCRNN2(MyModel):
    def __init__(
        self, num_classes, shape=(20, 112, 112, 3), nbout=2, momentum=0.9
    ):
        super().__init__(num_classes, shape, nbout, momentum)

    def define_model(self):
        # CNN part

        model = tf.keras.Sequential()

        # CNN part here
        model.add(
            tf.keras.layers.experimental.preprocessing.RandomRotation(
                (-0.3, 0.3)
            )
        )

        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=self.shape,
                padding="same",
                activation="relu",
            ),
        )
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=self.shape,
                padding="same",
                activation="relu",
            )
        )
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(MaxPool2D(strides=(2, 2)))

        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=self.shape,
                padding="same",
                activation="relu",
            )
        )
        model.add(BatchNormalization(momentum=self.momentum))
        model.add(MaxPool2D(strides=(2, 2)))

        # model.add(GlobalMaxPool2D())

        # here, you can also use GRU or LSTM, RNN
        model.add(GRU(256, activation="relu", return_sequences=True))
        model.add(GRU(256, activation="relu"))

        # and finally, we make a decision network, Dense layer
        model.add(Dense(1024, activation="relu"))

        model.add(Dropout(0.5))

        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation="relu"))

        # output layers
        model.add(Dense(self.nbout, activation="softmax"))

        return model


class ModelConvet(MyModel):
    def __init__(
        self, num_classes, shape=(20, 112, 112, 3), nbout=2, momentum=0.9
    ):
        super().__init__(num_classes, shape, nbout, momentum)

    def define_model(self):
        model = tf.keras.Sequential()

        # resize picture for reducing complexicity
        model.add(
            Conv2D(
                32,
                (3, 3),
                input_shape=self.shape,
                padding="same",
                activation="relu",
            )
        )

        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(momentum=self.momentum))

        model.add(MaxPool2D(strides=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(momentum=self.momentum))

        model.add(MaxPool2D(strides=(2, 2)))

        model.add(GlobalMaxPool2D())

        return model


class ModelAction(MyModel):
    def __init__(
        self, num_classes, shape=(5, 112, 112, 3), nbout=2, momentum=0.9
    ):
        super().__init__(num_classes, shape, nbout, momentum)

    def define_model(self):
        # part 2 and 3: 2 GRU layers and 4 dense layers
        # Create our convnet
        convnet = ModelConvet(self.shape[1:])

        # then create our final model
        model = tf.keras.Sequential()

        # add the convnet
        model.add(TimeDistributed(convnet.model, input_shape=self.shape))

        # here, you can also use GRU or LSTM
        model.add(GRU(256, activation="relu", return_sequences=True))
        model.add(GRU(256, activation="relu"))

        # and finally, we make a decision network
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(self.nbout, activation="softmax"))
        return model
