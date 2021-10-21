# TensorFlow and tf.keras
import csv
import logging
import os
import os.path
import re
import shutil
from math import ceil, floor
from random import randint
from subprocess import run

import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    "G",
    "NG",
]


def video_to_frame_with_label(output_path, video_file, labels):
    """Dump video frames to image, then re-org folder based on label.

    Argument
    --------
      video_file: anything ffmpeg can handle.

      labels: list of dict.

        - name: string, name of the label data
        - label: CSV, three cols: start time, end time, duration. All
          in seconds. Frame index will be computed based on the time
          stamp.

    Return
    ------
      path: resulting folder for tf to load
    """

    # image filename convention
    FRAME_IMAGE_FILENAME = os.path.join(output_path, """%d.png""")
    FILE_PATTERN = re.compile(r"\d+(?=.png)")

    # load video info
    probe = ffmpeg.probe(video_file)
    video_stream = next(
        (
            stream
            for stream in probe["streams"]
            if stream["codec_type"] == "video"
        ),
        None,
    )
    frame_rate = video_stream.get("r_frame_rate", "").split("/")
    if len(frame_rate) > 1:
        frame_rate = int(frame_rate[0])
    else:
        frame_rate = None
        logger.exception("Video has unknown frame rate. Abort.")
        return

    # extract all frames from video
    run(["ffmpeg", "-i", video_file, FRAME_IMAGE_FILENAME])

    # get list of all images just extracted
    all_images = set(
        [name for name in os.listdir(output_path) if FILE_PATTERN.match(name)]
    )

    # organize images based on tags
    for label in labels:
        # group tagged vs. others
        tagged_images, untagged_images = categorize_images(
            all_images, label["name"], label["tags"], frame_rate
        )

        # move tagged and untagged to its own folders
        reorg_image_files_on_disk(
            output_path, label["name"], tagged_images, untagged_images
        )

    # delete all frames
    for i in all_images:
        os.remove(os.path.join(output_path, i))


def categorize_images(images, category_name, tag_timestamps, frame_rate):
    """Categorize images based on tags.

    Argument
    --------

      images: [string], list of image names
      category_name: string, as name
      tag_timestamps: csv file, has tag info
      frame_rate: int, convert time stamp to frame index

    Return
    ------

      tagged_images: set, image file names matching category tag
      untagged_images: set, non-tagged file names

    """

    # load tag timestamps
    tagged = []
    with open(tag_timestamps, newline="") as csvfile:
        reader = csv.DictReader(
            csvfile, fieldnames=["start", "end", "duration"]
        )

        # skip header line
        next(reader, None)

        # data
        tagged = [row for row in reader]

    tagged_images = []
    for t in tagged:
        start_index = floor(float(t["start"]) * frame_rate)
        end_index = ceil(float(t["end"]) * frame_rate)
        tagged_images += [
            "{}.png".format(x) for x in range(start_index, end_index + 1)
        ]

    untagged_images = images - set(tagged_images)
    return (set(tagged_images), untagged_images)


def reorg_image_files_on_disk(
    output_path, tag_name, tagged_images, untagged_images
):
    """Reorg image files on disk based on tag name.

    For example, if tag name is "grooming", we should expect two
    folders, `/grooming` and `non-grooming`. Each holds a list of
    image files according to the tagged_images and untagged_images
    list.

    Argument
    --------

      output_path: string, path to images
      tag_name: as name
      tagged_images: [string], list of file names. Name has no path.
      untagged_images: [string], list of file name. Name has no path.

    """
    # move file to tagged & untagged subfolders
    non_tagged_name = "non-{}".format(tag_name)

    for p in [tag_name, non_tagged_name]:
        target = os.path.join(output_path, p)
        if not os.path.exists(target):
            os.makedirs(target)

    # copy files
    tag_output_path = os.path.join(output_path, tag_name)
    for f in tagged_images:
        src = os.path.join(output_path, f)
        if os.path.exists(src):
            shutil.copy(src, tag_output_path)

    untagged_output_path = os.path.join(output_path, non_tagged_name)
    for f in untagged_images:
        src = os.path.join(output_path, f)
        if os.path.exists(src):
            shutil.copy(src, untagged_output_path)


def video_to_np_array(video_file):
    """Load training image & label, and testing image & label.

    Labels are categorization value w/ 1:1 mapping to the image.

    Test data is used to verify the model quality after
    training. Testing set can have different size from training.

    Return
    ------
      np array:
    """

    probe = ffmpeg.probe(video_file)
    video_stream = next(
        (
            stream
            for stream in probe["streams"]
            if stream["codec_type"] == "video"
        ),
        None,
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    # read video
    out, _ = (
        ffmpeg.input(video_file)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True)
    )

    # convert to np array
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return video


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
    video_to_frame_with_label(output_path, video_file, labels)

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
