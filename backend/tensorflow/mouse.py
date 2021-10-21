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
from video import VideoProcessor

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "G",
    "NG",
]


def load_data(path, subset, validation_split, batch_size, height, width):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(height, width),
        shuffle=True,
        seed=123,
        validation_split=validation_split,
        subset=subset,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )


def define_model():
    """Construct your AI layers.

    Pick optimizer, loss function & accuracy metrics are arts. This is
    the core decision of how your work might be different from others.

    """
    # from another tutorial
    num_classes = 2

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
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


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


def get_dataset(video_file, labels):

    output_path = os.path.split(video_file)[0]

    # video frame images
    video_processor = VideoProcessor(video_file, labels, output_path)
    video_processor.run()
    1/0

    # load images as dataset
    train_ds = load_data(
        output_path,
        subset="training",
        batch_size=32,
        validation_split=0.2,
        width=64,
        height=64,
    )
    val_ds = load_data(
        output_path,
        subset="validation",
        batch_size=32,
        validation_split=0.2,
        width=64,
        height=64,
    )
    class_names = train_ds.class_names
    print(class_names)

    # view some data
    # plot_dataset(train_ds, class_names, 9)

    return (output_path, train_ds, val_ds)


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
    EPOCHS = 10

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

    # load training data set
    print("loading data")
    output_path, train_ds, val_ds = get_dataset(TRAIN_VIDEO, TRAIN_LABELS)
    print(train_ds.take(1))

    # config dataset for performance
    print("config dataset for performance")
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # define model
    print("defining model")
    the_model = define_model()

    # train model
    print("training model")
    the_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # evaluate model w/ test sets
    print("evaluating model")
    test_loss, test_acc = the_model.evaluate(val_ds)
    print("Test accuracy: {}, Test lost: {}".format(test_acc, test_loss))

    # define prediction model
    print("define prediction model")
    probability_model = tf.keras.Sequential(
        [the_model, tf.keras.layers.Softmax()]
    )

    # load test images
    output_path, test_ds, val_ds = get_dataset(TEST_VIDEO, TEST_LABELS)

    # make a prediction
    print("making prediction")
    predictions = probability_model.predict(test_ds)

    # plot some image and its prediction for visual evaluation
    plot_loss()
    test_images = np.concatenate([x for x, y in test_ds], axis=0)
    test_labels = np.concatenate([y for x, y in test_ds], axis=0)
    plot_image_and_prediction(5, 2, predictions, test_images, test_labels)


if __name__ == "__main__":
    main()
