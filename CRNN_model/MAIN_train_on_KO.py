import gc
import os
import os.path

import numpy as np
import tensorflow as tf
from Script.CRNN import *
# from Script.Prepare_data import *
from Script.Prepare_data import get_train_xy, prepare_video_data_31

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# np.random.seed(12)
# folder paths
# store all videos in same folder
# store all grooming data in same folder
ROOT = os.getcwd()
data_folder = os.path.join(ROOT, "../data/train/train_data/")
video_folder = os.path.join(ROOT, "../data/train/train_video/")
save_name = os.path.join(ROOT, "../data/Best_model_31length.h5")
plot_name = os.path.join(ROOT, "../data/train/loss_and_acc_new_31length.png")

train_loss = []
train_acc = []
test_loss = []
test_acc = []

# decided based on loss and acc
EPOCHS = 100

# divided into 10 parts and 1 min each
gap = 1

# train on the videos in train folder
for i in range(len(os.listdir(data_folder))):
    # transfrom video to array and get labels
    # only load first 10 mins here (600 sec)

    all_videos, all_labels = get_train_xy(
        data_folder, video_folder, start=i, end=i + 1
    )

    all_videos = np.array(all_videos)
    all_labels = np.array(all_labels)
    print("Shape of test video: ", all_videos.shape)
    print("Shape of test labels: ", all_labels.shape)

    # train 10 mins videos
    mouse = sorted(os.listdir(data_folder))[i][:2]
    first_time = True

    for j in range(0, 10, gap):
        """
        #bootstrap = np.random.randint(9*1800)
        #print("Selected starting time: ", round(bootstrap/1800,3))
        #train_x, train_y = prepare_video_data_2(all_videos, all_labels, bootstrap, bootstrap+1800)
        """
        # train_x, train_y = prepare_video_data(all_videos, all_labels, j, j+gap)

        # for 31 sequence length
        train_x, train_y = prepare_video_data_31(
            all_videos, all_labels, j, j + gap
        )

        # check class weights
        counts_class = np.bincount(train_y)
        print("Number of classes in this clip: ", len(counts_class))

        if len(counts_class) != 2:
            print("No grooming data in this part of video!\n SKIP!!!\n")
            continue

        grooming_ratio = 1 / (counts_class[1]) * len(train_y) / 2
        non_grooming_ratio = 1 / (counts_class[0]) * len(train_y) / 2
        class_weight = {0: non_grooming_ratio, 1: grooming_ratio}
        print(
            "Ratio of Grooming and Non grooming time in this clip:\n",
            class_weight,
        )

        # oversampling on both grooming and non grooming data
        train_x, train_y = oversampling(train_x, train_y)
        print("Oversampling DONE!\n")

        # make train y fit in the model
        train_y = tf.keras.utils.to_categorical(train_y, 2)

        # for later runs, load the best model
        # only build model on first run

        if j != 0 and first_time == False:
            model = tf.keras.models.load_model(save_name)
        else:
            model = initial_model(train_x)
            first_time = False

        history = train_model_frame_by_frame(model, train_x, train_y, EPOCHS)
        del train_x
        del train_y
        print(j + gap, "mins done!\n")
        # print(j,"/30 done!")
        train_loss += history.history["loss"]
        train_acc += history.history["acc"]
        test_loss += history.history["val_loss"]
        test_acc += history.history["val_acc"]

        del history
        del model
        gc.collect()

    print(mouse, " TRAINED!\n")

# plot_name = "frame_by_frame_model/train/loss_and_acc_new.png"

plot_loss(train_loss, test_loss, train_acc, test_acc, fn=plot_name)
