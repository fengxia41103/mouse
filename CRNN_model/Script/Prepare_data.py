import os

import cv2
import numpy as np
import pandas as pd

np.random.seed(12)


def get_label(path, time=10):
    """
    Load labels file; in labels file, behavior are marked by second
    """

    # load label CSV
    labels = pd.read_csv(path)

    # filter by end time to `time` in minutes
    labels = labels[labels["End Time(s)"] <= 60 * time]

    # convert time in seconds to frame index, 30 frame/second
    start_frame = np.array(labels["Start Time(s)"].values * 30, dtype=int)
    end_frame = np.array(labels["End Time(s)"].values * 30, dtype=int)

    # make label
    result_labels = np.zeros(time * 1800)
    for i in range(len(start_frame)):
        j = start_frame[i]
        while j < end_frame[i]:
            result_labels[j] = 1
            j += 1

    counts = np.unique(result_labels, return_counts=True)
    grooming_time = round(labels["Duration(Second)"].sum(), 2)
    non_grooming_time = 600 - grooming_time
    print(
        "Grooming {:.2f} sec, Not Grooming: {:.2f} sec".format(
            grooming_time, non_grooming_time
        )
    )

    return result_labels


def get_video_data(path, limit=100):
    """Extract frame from video."""
    print("Start extracting frames from", path)

    cap = cv2.VideoCapture(path)
    count = 0
    all_video = []

    success = True
    while success and count < 1800 * limit:
        success, frame = cap.read()

        # gray color it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray = frame
        # percent by which the image is resized
        scale_percent = 20

        # calculate the 30 percent of original dimensions
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)

        # resize for quicker processing
        #width = height = 64
        output = cv2.resize(gray, (width, height))

        output = np.expand_dims(output, axis=2)

        all_video.append(output[15:-15, 30:-30])

        count += 1

        # if i % 1800 == 0:
        #   print("%.1f minutes processed." % (float(i) / 1800))

    print("Video loaded!")
    cap.release()
    cv2.destroyAllWindows()

    return all_video


def get_raw_video_data(path, limit=100):
    # path is direct file names, ex. '/../../video.avi'
    print("Start extracting frames from", path)
    # all_10mins = os.listdir(path)
    """
    for i in all_10mins:
        current_path = path+i+'/'
        current_video = os.listdir(current_path)
        temp = current_path + current_video[0]
    """
    cap = cv2.VideoCapture(path)
    i = 0
    all_video = []
    # limit = 1
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # gray = frame
        # percent by which the image is resized

        (thresh, blackAndWhiteImage) = cv2.threshold(
            gray, 117, 255, cv2.THRESH_BINARY_INV
        )

        scale_percent = 45

        # calculate the 30 percent of original dimensions
        width = int(blackAndWhiteImage.shape[1] * scale_percent / 100)
        height = int(blackAndWhiteImage.shape[0] * scale_percent / 100)
        # dsize
        dsize = (width, height)
        # resize image
        output = cv2.resize(blackAndWhiteImage, dsize)

        output = np.expand_dims(output, axis=2)

        all_video.append(output[20:-15, 70:-65])
        i += 1

        # if i % 1800 == 0:
        #   print("%.1f minutes processed." % (float(i) / 1800))

        # add stopping point
        if i >= 1800 * limit:
            break

    print("Video loaded!")
    cap.release()
    cv2.destroyAllWindows()

    all_video_ary = np.array(all_video)
    print("shape of the vidoe: ", all_video_ary.shape)

    return all_video_ary


def prepare_data(all_video, labels, sample_size=2000):

    # random sampling from all frames
    sample_idx = np.random.choice(
        np.arange(11, len(all_video) - 11), sample_size, replace=False
    )
    train_x = []
    train_y = []
    for i in sample_idx:
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (20 frames)
        left = i - 10
        right = i + 10

        temp_train = all_video[left:right]

        train_x.append(temp_train)
        if len(temp_train) != 20:
            print(i)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))

    return train_x, train_y


def prepare_data_31(all_video, labels, sample_size=2000):

    # random sampling from all frames
    sample_idx = np.random.choice(
        np.arange(16, len(all_video) - 16), sample_size, replace=False
    )
    train_x = []
    train_y = []
    for i in sample_idx:
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (20 frames)
        left = i - 15
        right = i + 15

        temp_train = all_video[left:right]

        train_x.append(temp_train)
        if len(temp_train) != 30:
            print(i)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))

    return train_x, train_y


def prepare_video_data(all_video, labels, start, end):
    # used in MAIN in trainging model
    # generate sequence as input
    train_x = []
    train_y = []
    for i in range((1800 * start) + 11, (1800 * end) - 11):
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (20 frames)
        left = i - 10
        right = i + 10

        temp_train = all_video[left:right]

        if len(temp_train) != 20:
            print(i)

        train_x.append(temp_train)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))

    return train_x, train_y


def prepare_video_data_31(all_video, labels, start, end):
    # used in MAIN in trainging model
    # generate sequence as input with length 31
    train_x = []
    train_y = []
    for i in range((1800 * start) + 16, (1800 * end) - 16):
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (20 frames)
        left = i - 15
        right = i + 15

        temp_train = all_video[left:right]

        if len(temp_train) != 30:
            print(i)

        train_x.append(temp_train)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))
    return train_x, train_y


def prepare_video_data_2(all_video, labels, start, end):
    # used in MAIN in trainging model
    # generate sequence as input for bootstrapping
    train_x = []
    train_y = []
    for i in range((start) + 11, (end) - 11):
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (20 frames)
        left = i - 10
        right = i + 10

        temp_train = all_video[left:right]

        if len(temp_train) != 20:
            print(i)

        train_x.append(temp_train)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))

    return train_x, train_y


def prepare_only_video_data(all_video, start, end, case):
    # no labels needed in this part (making labels for entire video)
    # change boundary when change sequence length
    if case == "start":
        train_x = []
        for i in range((1800 * start) + 10, (1800 * end)):
            # get training x (20 frames)
            left = i - 10
            right = i + 10

            temp_train = all_video[left:right]

            if len(temp_train) != 20:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    if case == "middle":
        train_x = []
        for i in range((1800 * start), (1800 * end)):
            # get training x (20 frames)
            left = i - 10
            right = i + 10

            temp_train = all_video[left:right]

            if len(temp_train) != 20:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    if case == "end":
        train_x = []
        for i in range((1800 * start), (1800 * end) - 11):
            # get training x (20 frames)
            left = i - 10
            right = i + 10

            temp_train = all_video[left:right]

            if len(temp_train) != 20:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    return train_x


def prepare_only_video_data_31(all_video, start, end, case):
    # no labels needed in this part (making labels for entire video)
    # change boundary when change sequence length
    if case == "start":
        train_x = []
        for i in range((1800 * start) + 15, (1800 * end)):
            # get training x (30 frames)
            left = i - 15
            right = i + 15

            temp_train = all_video[left:right]

            if len(temp_train) != 30:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    if case == "middle":
        train_x = []
        for i in range((1800 * start), (1800 * end)):
            # get training x (20 frames)
            left = i - 15
            right = i + 15

            temp_train = all_video[left:right]

            if len(temp_train) != 30:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    if case == "end":
        train_x = []
        for i in range((1800 * start), (1800 * end) - 16):
            # get training x (20 frames)
            left = i - 15
            right = i + 15

            temp_train = all_video[left:right]

            if len(temp_train) != 30:
                print(i)

            train_x.append(temp_train)

        train_x = np.array(train_x) / 255.0
        print("shape of training set: ", train_x.shape)

    return train_x


def half_half_sampling(all_video, labels, sample_size=10800, time=60, ratio=1):
    # find all grooming index and non grooming index
    # then random sampling on non grooming inedex
    # make grooming and non grooming training sets same size
    # use this list of index generate training set

    # get all grooming
    grooming = np.nonzero(labels)[0]
    grooming_idx = list(grooming[grooming < (1800 * time)])

    # print('number of all grooming data:')
    # print(len(grooming_idx))

    # get same amount of non grooming video
    n_non_grooming = int(len(grooming_idx) * (ratio))

    non_grooming = np.argwhere(labels == 0).T[0]
    non_grooming = non_grooming[non_grooming <= (1800 * time)]

    ng_idx = np.random.choice(
        np.arange(11, len(non_grooming) - 11), n_non_grooming, replace=False
    )
    # print('number of all non grooming data:')
    # print(len(ng_idx))

    all_idx = np.concatenate((grooming_idx, ng_idx), axis=0)
    np.random.shuffle(all_idx)

    # print("Range of sampling: ")
    # print(all_idx.shape[0])
    sample_idx = np.random.choice(
        np.arange(all_idx.shape[0]), sample_size, replace=False
    )

    print("total sample size: ")
    print(len(sample_idx))

    # generate data
    train_x = []
    train_y = []
    for j in sample_idx:
        i = all_idx[j]
        # get center label and make training y
        train_y.append(labels[i])

        # get training x (2*n  frames)
        left = i - 15
        right = i + 15

        temp_train = all_video[left:right]
        if len(temp_train) != 30:
            print(i)

        train_x.append(temp_train)

    train_x = np.array(train_x) / 255.0
    print("shape of training set: ", train_x.shape)
    print("length of training set labels: ", len(train_y))

    return train_x, train_y


def listfolders(path):
    return sorted([f for f in os.listdir(path) if not f.startswith(".")])


def get_all_labels(folder_path, fn):

    temp_path = folder_path + fn[0]
    all_labels = get_label(temp_path, time=10)

    for i in range(1, len(fn)):
        temp_path = folder_path + fn[i]
        temp_labels = get_label(temp_path, time=10)

        all_labels = np.concatenate((all_labels, temp_labels))

    return all_labels


def get_all_video(folder, fn, limit=10):

    temp_path = folder + fn[0]

    videos = get_video_data(temp_path, limit=limit)

    for i in range(1, len(fn)):

        temp_path = folder + fn[i]
        temp_video = get_video_data(temp_path, limit=limit)

        videos = np.concatenate((videos, temp_video))

    return videos


def get_all_raw_video(folder, fn, limit=10):

    temp_path = folder + fn[0]

    videos = get_raw_video_data(temp_path, limit=limit)

    for i in range(1, len(fn)):

        temp_path = folder + fn[i]
        temp_video = get_raw_video_data(temp_path, limit=limit)

        videos = np.concatenate((videos, temp_video))

    return videos


def get_sets(all_video, labels, sample_size=2000, time=10):

    x, y = half_half_sampling(all_video, labels, sample_size, time)

    return x, y


def get_train_xy(data_folder, video_folder, start=0, end=3):
    data_fn = listfolders(data_folder)

    video_fn = listfolders(video_folder)

    print("Current data files:", data_fn[start:end])
    print("Current vdieo files:", video_fn[start:end])
    print("\n")

    labels = get_all_labels(data_folder, data_fn[start:end])
    videos = get_all_video(video_folder, video_fn[start:end])

    return videos, labels


def get_raw_train_xy(data_folder, video_folder, start=0, end=3):
    data_fn = listfolders(data_folder)

    video_fn = listfolders(video_folder)

    print("Current data files:", data_fn[start:end])
    print("Current vdieo files:", video_fn[start:end])
    print("\n")

    labels = get_all_labels(data_folder, data_fn[start:end])
    videos = get_all_raw_video(video_folder, video_fn[start:end])

    return videos, labels


def get_train_x(video_folder, start=0, end=3, limit=100):
    video_fn = listfolders(video_folder)
    videos = get_all_video(video_folder, video_fn[start:end], limit=limit)

    return videos


def get_raw_train_x(video_folder, start=0, end=3, limit=100):
    video_fn = listfolders(video_folder)
    videos = get_all_raw_video(video_folder, video_fn[start:end], limit=limit)

    return videos


if __name__ == "__main__":

    labels = get_label("../data/KO_data/A2_Groom.csv", time=10)

    all_video = get_video_data("../video/KO_model/A2#_10min_black.mp4")
    # all_video = get_raw_video_data('../raw_10mins/')

    # train_x, train_y = prepare_data(all_video, labels, sample_size = 200)
    train_x, train_y = half_half_sampling(
        all_video, labels, sample_size=2000, time=10
    )

    print("Trianing set info:")
    print(np.unique(train_y, return_counts=True, axis=0))
