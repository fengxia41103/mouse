# TensorFlow and tf.keras
from random import randint

import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_training_and_testing_data():
    """Load training image & label, and testing image & label.

    Labels are categorization value w/ 1:1 mapping to the image.

    Test data is used to verify the model quality after
    training. Testing set can have different size from training.

    Return
    ------

    train_images: training image, array of 28x28 image data.
    train_labels: values of 0-9 representing 10 categories of images.
    test_images: testing image set.
    test_labels: testing image categorization values.
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist

    return fashion_mnist.load_data()


def define_model():
    """Construct your AI layers.

    Pick optimizer, loss function & accuracy metrics are arts. This is
    the core decision of how your work might be different from others.

    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


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

    colors = {"wrong predictions": "gray", 'predicted "actual"': "blue"}
    labels = list(colors.keys())
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels
    ]
    plt.legend(handles, labels)


def plot_image_and_prediction(
    num_rows, num_cols, predictions_array, test_images, test_labels
):

    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        index = randint(0, len(test_labels))
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions_array[index], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions_array[index], test_labels)
    plt.tight_layout()
    plt.show()


def main():

    # load data
    print("loading data")
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = load_training_and_testing_data()

    # validate array size are 1:1 mapping
    assert train_images.shape[0] == len(train_labels)
    assert test_images.shape[0] == len(test_labels)

    # define model
    print("defining model")
    the_model = define_model()

    # train model
    # TODO: magic number
    print("training model")
    the_model.fit(train_images, train_labels, epochs=3)

    # evaluate model w/ test sets
    print("evaluating model")
    test_loss, test_acc = the_model.evaluate(test_images, test_labels)
    print("\nTest accuracy:", test_acc)

    # define prediction model
    print("define prediction model")
    probability_model = tf.keras.Sequential(
        [the_model, tf.keras.layers.Softmax()]
    )

    # make a prediction
    predictions = probability_model.predict(test_images)

    # plot some image and its prediction for visual evaluation
    plot_image_and_prediction(5, 3, predictions, test_images, test_labels)


if __name__ == "__main__":
    main()
