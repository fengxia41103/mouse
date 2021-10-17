from Prepare_data import *
import os
import pandas as pd
import numpy as np

def listfolders(path):
    return sorted([f for f in os.listdir(path) if not f.startswith('.')])

def get_all_labels(folder_path, fn):
    
    temp_path = folder_path + fn[0]
    all_labels = get_label(temp_path, time = 10)
    
    for i in range(1, len(fn)):
        temp_path = folder_path + fn[i]
        temp_labels = get_label(temp_path, time = 10)
        
        all_labels = np.concatenate((all_labels, temp_labels))
        
    return all_labels
    
    
def get_all_video(folder, fn):
    
    temp_path = folder + fn[0]
   
    videos = get_video_data(temp_path, limit = 10)
    
    for i in range(1, len(fn)):
        
        temp_path = folder + fn[i]
        temp_video = get_video_data(temp_path, limit = 10)
        
        videos = np.concatenate((videos, temp_video))
        
    return videos
        
    
def get_sets(all_video, labels, sample_size = 2000, time = 10):
    
    x,y = half_half_sampling(all_video, labels, sample_size, time)
    
    return x,y 


def get_train_xy(data_folder, video_folder, start = 0, end = 3):
    data_fn = listfolders(data_folder)
    
    video_fn = listfolders(video_folder)
    
    labels = get_all_labels(data_folder, data_fn[start:end])
    videos = get_all_video(video_folder, video_fn[start:end])
    
    return videos, labels
    
if __name__ == "__main__":
    
    data_folder = "../data/KO_data/"
    
    video_folder = "../video/KO_model/"
    
    videos, labels = get_train_xy(data_folder, video_folder, start = 0, end = 2)
    
    x,y = get_sets(videos, labels, sample_size = 900*4, time = 10)
    
    print("Trianing set info:")
    print(np.unique(y, return_counts = True, axis = 0))
    
    