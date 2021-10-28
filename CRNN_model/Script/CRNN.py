#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalMaxPool2D,
    MaxPool2D,
    TimeDistributed,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling, Resizing

# In[ ]:


def build_convnet(shape):
    # part 1: CNN block: 3 layers of CNN
    momentum = 0.9
    model = tf.keras.Sequential()

    # resize picture for reducing complexicity
    model.add(
        Conv2D(32, (3, 3), input_shape=shape, padding="same", activation="relu")
    )

    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D(strides=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization(momentum=momentum))

    model.add(MaxPool2D(strides=(2, 2)))

    model.add(GlobalMaxPool2D())

    return model


def action_model(shape=(5, 112, 112, 3), nbout=2):
    # part 2 and 3: 2 GRU layers and 4 dense layers
    # Create our convnet
    convnet = build_convnet(shape[1:])

    # then create our final model
    model = tf.keras.Sequential()
    # add the convnet
    model.add(TimeDistributed(convnet, input_shape=shape))

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
    model.add(Dense(nbout, activation="softmax"))
    return model


def crnn_model(shape=(20, 112, 112, 3), nbout=2):

    # CNN part
    momentum = 0.9

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
                32, (3, 3), input_shape=shape, padding="same", activation="relu"
            ),
            input_shape=shape,
        )
    )
    model.add(TimeDistributed(MaxPool2D(strides=(2, 2))))

    model.add(
        TimeDistributed(
            Conv2D(
                32, (3, 3), input_shape=shape, padding="same", activation="relu"
            )
        )
    )
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
    model.add(TimeDistributed(MaxPool2D(strides=(2, 2))))

    model.add(
        TimeDistributed(
            Conv2D(
                32, (3, 3), input_shape=shape, padding="same", activation="relu"
            )
        )
    )
    model.add(TimeDistributed(BatchNormalization(momentum=momentum)))
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
    model.add(Dense(nbout, activation="softmax"))

    return model


def plot_result(history, fn="CRNN.png"):

    fig = plt.figure(figsize=(15, 5))
    # accuracy
    plt.subplot(121)
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")

    # loss
    plt.subplot(122)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper right")
    plt.show()

    fig.savefig(fn)


def plot_loss(
    train_loss, test_loss, train_acc, test_acc, fn="loss_and_acc.png"
):
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(train_loss)
    plt.plot(test_loss)

    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper right")

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(test_acc)

    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="lower left")

    plt.tight_layout()
    fig.savefig(fn)


def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:

        y_pred = y_pred[:, 1:2]
        y_true = y_true[:, 1:2]
    return y_true, y_pred


def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def initial_model(train_x):
    # build the model
    SIZE = train_x[0][0].shape[:-1]
    CHANNELS = 1
    NBFRAME = 20
    BS = 10
    classes = [0, 1]

    INSHAPE = (NBFRAME,) + SIZE + (CHANNELS,)  # (21, width, length, 1)

    # strategy = tf.distribute.MirroredStrategy(["GPU:1"])
    strategy = tf.distribute.MirroredStrategy()

    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = crnn_model(INSHAPE, len(classes))
        optimizer = tf.keras.optimizers.Adam(0.001)  # can try different
        model.compile(
            optimizer,
            "categorical_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.TrueNegatives(name="TN"),
            ],
        )

    return model


def train_model(model, train_x, train_y, EPOCHS, mouse, model_fn):
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    # model_fn ='Each_mouse_model/' + mouse + '/Best_model.h5'
    # model_fn = 'frame_by_frame_model/train/models/Best_model_raw.h5'

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.ModelCheckpoint(
            model_fn,
            verbose=1,
            monitor="val_TN",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_TN", mode="max", patience=25, verbose=1
        ),
        MyCustomCallback(),
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_split=0.3,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return history


def oversampling(train_x, train_y):
    # oversampling the model; force two classes with equal weights
    counts_classes = np.bincount(train_y)
    search_key = [0, 1]
    index_0 = np.where(np.array(train_y) == search_key[0])[0]
    index_1 = np.where(np.array(train_y) == search_key[1])[0]

    # random sampling with replacement to balance data set
    print("random sampling with replacement to balance data set")
    choices_0 = np.random.choice(index_0, counts_classes[1])
    choices_1 = np.random.choice(index_1, counts_classes[0])

    print("get labels and data from sampling index")
    sampling_y_0 = np.array(train_y)[choices_0]
    sampling_y_1 = np.array(train_y)[choices_1]

    sampling_x_0 = train_x[choices_0]
    sampling_x_1 = train_x[choices_1]

    print("concate sampling set with original set")
    train_y = np.concatenate([sampling_y_0, np.array(train_y)])
    del sampling_y_0
    train_y = np.concatenate([sampling_y_1, np.array(train_y)])
    del sampling_y_1
    gc.collect()

    print("get train_x")
    train_x = np.concatenate([sampling_x_0, train_x])
    del sampling_x_0
    train_x = np.concatenate([sampling_x_1, train_x])
    del sampling_x_1
    gc.collect()

    return train_x, train_y


def train_model_frame_by_frame(model, train_x, train_y, EPOCHS):
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()

    # model_fn ='frame_by_frame_model/train/models/Best_model.h5'

    # model save here
    model_fn = "frame_by_frame_model/train/models/Best_model_31length.h5"

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.ModelCheckpoint(
            model_fn,
            verbose=1,
            monitor="val_TN",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_TN", mode="max", patience=25, verbose=1
        ),
        MyCustomCallback(),
    ]

    history = model.fit(
        train_x,
        train_y,
        validation_split=0.3,
        verbose=1,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    return history
