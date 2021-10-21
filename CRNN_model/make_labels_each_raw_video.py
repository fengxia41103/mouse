import gc
import os

import cv2
import numpy as np
import pandas as pd
from eval_model import *

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def make_labels(model, test_x, case):
    D1_final_prediction = []
    if case == "start":
        # no prediction at the begining
        model = tf.keras.models.load_model(model)

        D1_predicted = model.predict(test_x)

        D1_final_prediction += ["no_prediction"] * 10
        for i in D1_predicted:
            # if grooming prediction greater than 0.7 consider grooming
            if i[1] > 0.7:
                D1_final_prediction.append("Grooming")
            else:
                D1_final_prediction.append("Not Grooming")

    if case == "middle":
        # no no_prediction
        model = tf.keras.models.load_model(model)

        D1_predicted = model.predict(test_x)

        for i in D1_predicted:
            # if grooming prediction greater than 0.7 consider grooming
            if i[1] > 0.7:
                D1_final_prediction.append("Grooming")
            else:
                D1_final_prediction.append("Not Grooming")

    if case == "end":
        # no priection at the end
        model = tf.keras.models.load_model(model)

        D1_predicted = model.predict(test_x)

        for i in D1_predicted:
            # if grooming prediction greater than 0.7 consider grooming
            if i[1] > 0.7:
                D1_final_prediction.append("Grooming")
            else:
                D1_final_prediction.append("Not Grooming")

        D1_final_prediction += ["no_prediction"] * 11

    return D1_final_prediction


if __name__ == "__main__":

    # mouses = sorted(os.listdir('Each_mouse_raw_video/'))[:-1]
    mouses = ["B1", "C3", "D3", "D6"]
    print("Start evaluation on:")
    print(mouses)

    for mouse in mouses:
        # get training set
        data_folder = "Each_mouse_raw_video/" + mouse + "/data/"
        video_folder = "Each_mouse_raw_video/" + mouse + "/video/"

        # 4 specific folder names

        # load 100 min video
        all_videos = get_raw_train_x(video_folder, start=0, end=1, limit=100)
        all_videos = np.array(all_videos)

        print("Shape of test video: ", all_videos.shape)
        print("\n")

        slot = 1
        labels = []
        for j in range(0, int(100 / slot)):
            print("Preparing data\n")
            # make sure have labels between clips
            case = "middle"
            # modified from prepare_video_data, ignore lables this time
            if j == 0:
                case = "start"
            if j == int(100 / slot) - 1:
                case = "end"

            test_x = prepare_only_video_data(
                all_videos, j * slot, (j + 1) * slot, case
            )

            print("Making labels for ", mouse, "\n")
            best_model = "Each_mouse_raw_video/" + mouse + "/Best_model.h5"
            temp_labels = make_labels(best_model, test_x, case)

            labels += temp_labels
            print(len(temp_labels), "frames labeled!")
            print((j + 1) * slot, "mins done!\n")

            del test_x
            del temp_labels
            gc.collect()

        final_labels = pd.DataFrame(labels, columns=["Labels"])
        file_name = (
            "Each_mouse_raw_video/" + mouse + "/" + mouse + "_Labels_raw.csv"
        )
        final_labels.to_csv(file_name)
        print("Total {} frames saved!".format(len(labels)))
        print("finished!")
