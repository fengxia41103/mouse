# TensorFlow and tf.keras
import csv
import logging
import os
import os.path
from math import ceil
from math import floor
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mydata import DataLoader
from mymodel import ModelCRNN
from mymodel import ModelCRNN2
from mymodel import ModelType1
from myvideo import VideoProcessor

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "G",
    "NG",
]


def plot_dataset(ds, class_names, count=9):
    image_batch, label_batch = next(iter(ds))
    plt.figure(figsize=(10, 10))
    for i in range(count):
        plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        label = label_batch[i]
        plt.title(class_names[label])
        plt.axis("off")
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel(
        "{:2.0f}%: {}\nactual: {}".format(
            100 * np.max(predictions_array),
            CLASS_NAMES[predicted_label],
            CLASS_NAMES[true_label],
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_labels):
    true_label = true_labels[i]
    total = len(CLASS_NAMES)

    plt.grid(False)
    plt.xticks(range(total))
    plt.yticks([])
    thisplot = plt.bar(range(total), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    # prediction
    thisplot[predicted_label].set_color("gray")

    # actual
    thisplot[true_label].set_color("blue")

    # legend
    colors = {"wrong predictions": "gray", 'predicted "actual"': "blue"}
    labels = list(colors.keys())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels
    ]
    plt.legend(handles, labels)


def plot_image_and_prediction(
    num_rows, num_cols, predictions_array, test_images, test_labels
):
    """Visualize some test image and their prediction label.

    Argument
    --------

      num_rows: int, plot grid height
      num_cols: int, plot grid width
      prediction_array: TF predict return type
      test_images: array of image data as matrix
      test_labels: [int], list of image labels

    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        # randomly select image to print
        index = randint(0, len(test_labels))

        # show actual image
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions_array[index], test_labels, test_images)

        # show prediction bar chart
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions_array[index], test_labels)

    plt.tight_layout()
    plt.show()


def plot_loss(history):
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])

    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper right")

    # accuray of train & val
    plt.subplot(122)
    plt.plot(history["acc"])
    plt.plot(history["val_acc"])

    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="lower left")
    plt.tight_layout()
    fig.savefig("loss_and_acc.png")


def main():
    # training iteration
    EPOCHS = 1

    # configs
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    HEIGHT = 64
    WIDTH = 64

    # training video & tag timestamps
    TRAIN_VIDEO = "../../data/train/train_video/A2#_10min_black.mp4"
    TRAIN_LABELS = [
        {"name": "grooming", "tags": "../../data/train/train_data/A2_frame.csv"}
    ]

    # testing video & tag timestamps
    TEST_VIDEO = "../../data/test/test_video/B4_917_10min_black.mp4"
    TEST_LABELS = [
        {
            "name": "grooming",
            "tags": "../../data/test/test_data/B4_917_frame.csv",
        }
    ]

    logger.info("Load training video")
    output_path = os.path.split(TRAIN_VIDEO)[0]
    training_video = VideoProcessor(TRAIN_VIDEO, TRAIN_LABELS, output_path)
    training_video.run()

    # step 2: load up train & val ds
    training_data = DataLoader(
        training_video.output_path,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        width=WIDTH,
        height=HEIGHT,
    )
    training_data.run()

    # convert data array to fit model input shape
    # TBD


    logger.info("training model")
    # the_model = ModelType1(num_classes=len(TRAIN_LABELS) + 1)
    the_model = ModelCRNN2(
        num_classes=len(TRAIN_LABELS) + 1, shape=(1, WIDTH, HEIGHT, 3)
    )
    the_model.run(training_data.train_ds, training_data.val_ds, EPOCHS)
    logger.info(
        "Test accuracy: {}, Test lost: {}".format(
            the_model.evaluation_accuracy, the_model.evaluation_loss
        )
    )

    # plot some image and its prediction for visual evaluation
    # plot_loss(the_model.training_history)
    # print(the_model.training_history)

    # process videos & frames
    logger.info("load testing data")
    output_path = os.path.split(TEST_VIDEO)[0]
    test_video = VideoProcessor(TEST_VIDEO, TEST_LABELS, output_path)
    test_video.run()

    test_data = DataLoader(
        test_video.output_path,
        batch_size=BATCH_SIZE,
        validation_split=0.001,  # don't split test data
        width=WIDTH,
        height=HEIGHT,
    )
    test_data.run()

    # make a prediction
    logger.info("making prediction")
    the_model.predict(test_data.train_ds)

    # confusion matrix
    logger.info("build confusion matrix")
    confusion_matrix = tf.math.confusion_matrix(
        test_data.train_labels,
        the_model.predictions_single,
        num_classes=test_data.num_of_classes,
        weights=None,
        dtype=tf.dtypes.int32,
        name=None,
    )

    # if = 1: [[  321 11359][  189  6113]]
    # if = 3: [[  415 11265][  220  6082]]
    print(confusion_matrix)

    # show some visual
    plot_image_and_prediction(
        5,
        2,
        the_model.predictions,
        test_data.train_images,
        test_data.train_labels,
    )


if __name__ == "__main__":
    main()
