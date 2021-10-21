import gc
import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from Script.Prepare_data import *
from sklearn.metrics import confusion_matrix

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


np.random.seed(12)


def evlu_model(model, test_x, test_y):
    model = tf.keras.models.load_model(model)

    scores = model.evaluate(test_x, test_y, verbose=0)
    print("{}: {:.2%}".format(model.metrics_names[1], scores[1]))
    del scores
    gc.collect()

    # final results(p0, p1)
    D1_predicted = model.predict(test_x)

    D1_final_prediction = []
    for i in D1_predicted:
        # if grooming prediction greater than 0.7 consider grooming
        # compare two directly
        if i[1] >= i[0]:
            D1_final_prediction.append(1)
        else:
            D1_final_prediction.append(0)

    D1_final_prediction = np.array(D1_final_prediction)

    return D1_final_prediction


def plot_result(pred, true, fn):
    print("Predicted counts: ")
    print(np.unique(pred, return_counts=True))
    print("True counts: ")
    print(np.unique(true, return_counts=True, axis=0))

    print("Confusion Matrix: ")
    print(confusion_matrix(true, pred))

    predict_g = np.argwhere(pred == 1).T[0]
    true_g = np.argwhere(np.array(true) == 1).T[0]

    fig = plt.figure(figsize=(50, 5))
    plt.scatter(
        predict_g,
        np.full(len(predict_g), 0.05),
        marker="|",
        s=9000,
    )
    plt.scatter(
        true_g,
        np.zeros(len(true_g)),
        marker="|",
        s=9000,
        color="red",
    )

    blue = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="|",
        linestyle="None",
        markersize=10,
        label="Predticted",
    )
    red = mlines.Line2D(
        [],
        [],
        color="red",
        marker="|",
        linestyle="None",
        markersize=10,
        label="True",
    )

    plt.legend(handles=[blue, red])

    plt.ylim(-0.1, 0.2)
    save = "frame_by_frame_model/compare_31_length.png"
    fig.savefig(save)

    del fig, blue, red
    gc.collect()
    print("Plot 1 done!")
    return confusion_matrix(true, pred)


def pie_plot(cm, fn):
    labels = "True Postive", "False Negative", "False Postive", "True Negative"
    sizes = cm
    explode = (0, 0.1, 0.3, 0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        shadow=True,
        startangle=90,
    )
    ax1.axis(
        "equal"
    )  # Equal aspect ratio ensures that pie is drawn as a circle.
    save = "frame_by_frame_model/test/pie_plots/" + fn + "_pie_plot.png"
    fig1.savefig(save)
    print("Plot 2 done!")


if __name__ == "__main__":
    data_folder = "frame_by_frame_model/test/test_data/"
    video_folder = "frame_by_frame_model/test/test_video_100min/"
    # mouses = ["A2","A6","A7","B2","B4_917", "B5", "B6", "D2",
    #         "B1","B4_88", "C3", "D3","D6", "D7", "D8"]
    mouses = sorted(os.listdir(data_folder))
    print("Start testing on:")
    print(mouses)

    all_cm = []
    for i in range(len(mouses)):
        # get training set

        all_videos, all_labels = get_train_xy(
            data_folder, video_folder, start=i, end=i + 1
        )

        all_videos = np.array(all_videos)
        all_labels = np.array(all_labels)
        print("Shape of train labels: ", all_labels.shape)

        # test_x, test_y = half_half_sampling(all_videos, all_labels, sample_size = 900*8, time = 10, ratio = 1)
        # test_x, test_y = prepare_data(all_videos, all_labels, sample_size = 900*8)
        test_x, test_y = prepare_data_31(
            all_videos, all_labels, sample_size=900 * 8
        )

        test_y_1d = test_y

        del all_videos
        del all_labels
        gc.collect()

        test_y = tf.keras.utils.to_categorical(test_y, 2)
        # best_model = 'Each_mouse_model/D3/models/Best_model_2mins.h5'
        # best_model = 'frame_by_frame_model/train/models/Best_model.h5'
        best_model = "frame_by_frame_model/train/models/Best_model_31length.h5"

        D1_final_pred = evlu_model(best_model, test_x, test_y)
        del test_x, test_y
        gc.collect()

        cm = plot_result(D1_final_pred, test_y_1d, fn=mouses[i])
        del test_y_1d
        del D1_final_pred
        gc.collect()

        pie_plot(cm.flatten(), fn=mouses[i])

        all_cm.append(cm.flatten())
        print("Current mouse done!\n")

    all_mice_cm = pd.DataFrame(
        all_cm, index=mouses, columns=["TP", "FN", "FP", "TN"]
    )
    # save = 'frame_by_frame_model/test/comfusion_matirx.csv'
    save = "frame_by_frame_model/test/comfusion_matirx_31length.csv"
    all_mice_cm.to_csv(save, index=True)
