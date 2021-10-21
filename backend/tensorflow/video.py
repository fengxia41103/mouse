import csv
import logging
import os
import os.path
import re
import shutil
from math import ceil
from math import floor
from random import randint
from subprocess import run

import ffmpeg
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, video_file, labels, output_path="."):
        self.video = video_file
        self.labels = labels
        self.output_path = output_path
        self.all_images = []

        self.image_filename_convention = os.path.join(
            self.output_path, """%d.png"""
        )
        self.image_filename_pat = re.compile(r"\d+(?=.png)")

        # probe video
        probe = ffmpeg.probe(self.video)
        video_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )
        self.frame_rate = int(
            video_stream.get("r_frame_rate", "").split("/")[0]
        )
        self.width = int(video_stream["width"])
        self.height = int(video_stream["height"])

    def run(self):
        self.dump_video_to_frame()
        self._org_images_by_label()
        self._remove_frame_dumps()

    def dump_video_to_frame(self):
        """Dump video frames to image."""
        run(["ffmpeg", "-i", self.video, self.image_filename_convention])

        # get list of all images just extracted
        self.all_images = set(
            [
                name
                for name in os.listdir(self.output_path)
                if self.image_filename_pat.match(name)
            ]
        )

    def _categorize_images(self, category_name, tag_timestamps):
        """Categorize images based on tags.

        Argument
        --------

          category_name: string, as name
          tag_timestamps: csv file, has tag info

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
            start_index = floor(float(t["start"]) * self.frame_rate)
            end_index = ceil(float(t["end"]) * self.frame_rate)
            tagged_images += [
                "{}.png".format(x) for x in range(start_index, end_index + 1)
            ]

        untagged_images = self.all_images - set(tagged_images)
        return (set(tagged_images), untagged_images)

    def _reorg_image_files_on_disk(
        self, tag_name, tagged_images, untagged_images
    ):
        """Reorg image files on disk based on tag name.

        For example, if tag name is "grooming", we should expect two
        folders, `/grooming` and `non-grooming`. Each holds a list of
        image files according to the tagged_images and untagged_images
        list.

        Argument
        --------

          tag_name: as name
          tagged_images: [string], list of file names. Name has no path.
          untagged_images: [string], list of file name. Name has no path.

        """
        # move file to tagged & untagged subfolders
        non_tagged_name = "non-{}".format(tag_name)

        for p in [tag_name, non_tagged_name]:
            target = os.path.join(self.output_path, p)
            if not os.path.exists(target):
                os.makedirs(target)

        # move files
        tag_output_path = os.path.join(self.output_path, tag_name)
        for f in tagged_images:
            src = os.path.join(self.output_path, f)
            if os.path.exists(src):
                shutil.copy(src, tag_output_path)

        untagged_output_path = os.path.join(self.output_path, non_tagged_name)
        for f in untagged_images:
            src = os.path.join(self.output_path, f)
            if os.path.exists(src):
                shutil.copy(src, untagged_output_path)

    def _org_images_by_label(self):
        """Organize image based tag/label info.

        Tag file will tell us which image is tagged. The tag will be
        create as a folder, and its files will be moved into this
        folder. Once this is constructed, TF can load all data w/ one
        call.

        Labels can be an array. Each label corresponds to a
        behavior/tag we want to track. Each label has:

        1. name: the behavior name
        2. label data file: some sort of data we use to know which
           image has this behavior, thus we can group the image accordingly.
        """

        # organize images based on tags
        for label in self.labels:
            # group tagged vs. others
            tagged_images, untagged_images = self._categorize_images(
                label["name"], label["tags"]
            )

            # move tagged and untagged to its own folders
            self._reorg_image_files_on_disk(
                label["name"], tagged_images, untagged_images
            )

    def _remove_frame_dumps(self):
        # delete all frames
        for i in self.all_images:
            os.remove(os.path.join(self.output_path, i))

    def video_to_np_array(self):
        """Load training image & label, and testing image & label.

        Labels are categorization value w/ 1:1 mapping to the image.

        Test data is used to verify the model quality after
        training. Testing set can have different size from training.

        Return
        ------
          np array:
        """

        # read video
        out, _ = (
            ffmpeg.input(self.video)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True)
        )

        # convert to np array
        video = np.frombuffer(out, np.uint8).reshape(
            [-1, self.height, self.width, 3])
        return video
