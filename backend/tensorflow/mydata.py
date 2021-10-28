import numpy as np
import tensorflow as tf


class DataLoader:
    def __init__(
        self, working_path, batch_size, validation_split, width, height
    ):
        self.working_path = working_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.width = width
        self.height = height
        self.class_names = None
        self.num_of_classes = None

        # TF datasets
        self.train_ds = None
        self.train_images = None
        self.train_labels = None

        self.val_ds = None
        self.val_images = None
        self.val_labels = None

    def run(self):
        self._load_train_val_ds()
        self._tune_for_performance()

    def _load_data(self, subset):
        return tf.keras.utils.image_dataset_from_directory(
            self.working_path,
            labels="inferred",
            label_mode="int",
            class_names=None,
            color_mode="rgb",
            batch_size=self.batch_size,
            image_size=(self.height, self.width),
            shuffle=True,
            seed=123,
            validation_split=self.validation_split,
            subset=subset,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False,
        )

    def _load_train_val_ds(self):
        # load images as dataset
        self.train_ds = self._load_data(
            subset="training",
        )

        # set class names
        self.class_names = self.train_ds.class_names
        self.num_of_classes = len(self.class_names)

        # extract dedicated image array & label array
        self.train_images = np.concatenate(
            [x for x, y in self.train_ds], axis=0)
        self.train_labels = np.concatenate(
            [y for x, y in self.train_ds], axis=0)

        # same for validation data set
        self.val_ds = self._load_data(
            subset="validation",
        )
        self.val_images = np.concatenate([x for x, y in self.val_ds], axis=0)
        self.val_labels = np.concatenate([y for x, y in self.val_ds], axis=0)

    def _tune_for_performance(self):
        # config dataset for performance
        print("config dataset for performance")
        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = (
            self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        )
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
