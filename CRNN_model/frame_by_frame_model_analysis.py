import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest

def calculate_time(path):
    # load data, 
    # format: frame #, label
    # only load two classes here
    temp_df = pd.read_csv(path, index_col = 0)
    grooming = len(temp_df[temp_df["Labels"] == "Grooming"].values)
    non_grooming = len(temp_df[temp_df["Labels"] == "Not Grooming"].values)
    
    return grooming, non_grooming

def get_true_grooming(path):
    # load hand labeled data,
    # data format: starting time, end time, duration (in second)
    # output: list
    hand = pd.read_csv(path)
    # convert back to frame
    start = hand["Start Time(s)"].values *30
    duration = hand["Duration(Second)"].values

    hand_label = []
    for i in range(len(start)):
        temp = duration[i]*30
        j = 0
        while temp > j:
            # append frame# to the list to indicate true grooming time
            hand_label.append(int(start[i]+j))
            j += 1
    return hand_label

def filter_out_short(grooming):
    # not in used right now
    # used to combine close segments and remove minor segments
    pervious = 0
    temp_seq = []
    filtered = []
    for i in grooming.index.values:
        gap = i - pervious
        #print(gap)
        if gap > 5:
            # gap too big 
            if len(temp_seq) >= 10:
                # keep it in the result:
                filtered += temp_seq
                # considering drop minor segments
                # ex. total length < 10 frames
            temp_seq = []
        else:
            #if gap <= 5:
            # combine two segments(add node)
            temp_seq.append(i)
            #print(temp_seq)
        pervious = i
    return filtered

def plot_grooming(hand_label, predict_path, tilte, filter_out = False, time = 1):
    # compare true grooming and model predicted grooming 
    # output: matplotlib figure, predicted grooming time 
    temp_df = pd.read_csv(predict_path, index_col = 0)
    grooming = temp_df[temp_df["Labels"] == "Grooming"]
    fig = plt.figure(figsize = (10,2))
    
    if filter_out == True:
        grooming = filter_out_short(grooming)
        
        plt.scatter(grooming, np.full(len(grooming), 1)
                    ,marker = "|", s = 20000, alpha = 0.3)
        plt.xticks(np.arange(0,180000,1800*1), np.arange(0,100, 1))
        plt.scatter(hand_label, np.full(len(hand_label), 1.1)
                    ,marker = "|", s = 20000, alpha = 0.3)
        
    else:
    
        plt.scatter(grooming.index[:18000], np.full(len(grooming.index[:18000]), 1)
                    ,marker = "|", s = 20000, alpha = 0.3)
        plt.xticks(np.arange(0,180000,1800*1), np.arange(0,100, 1))
        plt.scatter(hand_label, np.full(len(hand_label), 1.1)
                    ,marker = "|", s = 20000, alpha = 0.3)
    plt.title(title, fontsize = 20)
    
    plt.xlim(-900+18000*(time-1),1800*10*time)
    plt.grid()
    return fig, grooming

def plot_cm(path, fn ='ConfustionMatrix_test.png'):
    # visualize comfusion matrix, based on grooming prediction 
    cm_data_test = pd.read_csv(path, index_col = 0)

    f = plt.figure(figsize=(12,8))
    ax = f.add_subplot(111)
    ind = np.arange(len(cm_data_test))
    width = 0.2

    rects2 = ax.bar(ind, cm_data_test["TN"], 2*width, color='seagreen')
    rects3 = ax.bar(ind+width, cm_data_test["FP"], width, color='red')
    rects4 = ax.bar(ind+2*width, cm_data_test["FN"], width, color='purple')
    ax.set_ylabel('Counts')
    ax.set_title('Confusion Matrix for Grooming Prediction', fontsize = 20)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(cm_data_test.index)
    ax.legend( (rects2[0],rects3[0], rects4[0]), 
              ('correct prediction', "missed prediction",'wrong prediction') )
    plt.show()
    f.savefig(fn)
    
    return cm_data_test

def calcualte_performence(cm_data):
    # return lists, with performence measurments for each mouse 
    TP = cm_data["TN"].values
    TN = cm_data["TP"].values
    FP = cm_data["FN"].values
    FN = cm_data["FP"].values

    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    Precision = TP/(TP+FP)
    #print(Precision)
    Recall = TP/(TP+FN)
    #print(Recall)
    F1 = 2*(Precision*Recall)/(Precision + Recall)
    #print(F1)
    return Accuracy, Precision, Recall, F1
    
def plot_performence(cm_data, Accuracy, Precision, Recall, F1, fn ="Permormance.png"):
    # visualize performence measurments
    # save graphs in given file name
    f = plt.figure(figsize=(15,9))
    ax = f.add_subplot(111)
    ind = np.arange(len(Accuracy))
    width = 0.2
    rects1 = ax.bar(ind-width, Accuracy, width, color = 'orange')
    rects2 = ax.bar(ind, Precision, width, color='seagreen')
    rects3 = ax.bar(ind+width, Recall, width, color='royalblue')
    rects4 = ax.bar(ind+2*width, F1, width, color='red')
    ax.set_ylabel('Counts')
    ax.set_title('Performance Measurements', fontsize = 20)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(cm_data.index)
    ax.legend( (rects1[0], rects2[0],rects3[0], rects4[0]), 
              ('Accuracy','Precision', "Recall",'F1') )
    plt.show()
    f.savefig(fn)

def model_performence(cm_data, fn):
    # first calculate performance measurements and make plots
    # then, calculate average performence measurments
    # no return, print only
    print("Precision: Percentage of grooming predictied correct among all grooming prediction.")
    print("Recall: Percentage of grooming predictied correct among all real grooming frames .")
    _ = calcualte_performence(cm_data)
    plot_performence(cm_data, _[0],_[1], _[2], _[3], fn)
    mean_accuracy = (cm_data["TP"].values + cm_data["TN"].values)/\
                    (cm_data["TP"].values + cm_data["TN"].values + 
                     cm_data["FP"].values + cm_data["FN"].values)
    mean_precision = cm_data["TN"].values/(cm_data["FN"].values + cm_data["TN"].values)
    mean_recall = cm_data["TN"].values/(cm_data["FP"].values + cm_data["TN"].values)
        
    print("Mean accuracy of the model: ", round(np.mean(mean_accuracy),3))
    print("Mean precision of the model: ", round(np.mean(mean_precision),3))
    print("Mean recall of the model: ", round(np.mean(mean_recall),3))
    
def two_sample_t_test(m1,m2,n1,n2,s1,s2):
    # 2 sample t-test, can use scipy.stats.ttest_ind instead
    sp = np.sqrt((((n1-1)*s1*s1) + ((n2-1)*s2*s2))/(n1+n2-2))
    t = (m1-m2)/(sp*np.sqrt(1/n1 + 1/n2))
    return t


def output_grooming_nongrooming(data,mouse_idx, fn = "Total_time_summary.csv"):
    # save total grooming time for all mice
    sum_of_WT = pd.DataFrame(index = mouse_idx,columns = ["Grooming","nonGrooming"])
    sum_of_WT["Grooming"] = data[0]
    sum_of_WT["nonGrooming"] = data[1]
    sum_of_WT.to_csv(fn)
    print("file saved in ", os.getcwd(),'/',fn)
    return sum_of_WT




if __name__ = main:
    '''
    for analysis difference between KO and WT
    run both group at one time 
    need seprate KO and WT before running 
    root file structure:
    |--root file
        |--hand_labels
           |--KO
              |--data.csv
           |--WT
              |--data.csv
        |--predicted_labels
           |--KO
              |--KO predict data.csv
           |--WT
              |--WT predict data.csv
           |--cm_KO.csv
           |--cm_WT.csv
    '''
    
    # first get the file path 
    root_path = ""
    os.chdir(root_path)
    # FIRST PART: each group analysis(KO, WT)
    # part 1.1 confusion matrix and performence measurements 
    # make plot for confusion matrix
    # only put testing(WT) data here
    cm_data_file_WT = "comfusion_matirx_test_new.csv"
    cm_plot_save_name_WT = 'ConfustionMatrix_test_new.png'
    cm_data_test_WT = plot_cm(cm_data_file_WT, cm_plot_save_name_WT)
    
    # training(KO) data here
    cm_data_file_KO = "comfusion_matirx_train_new.csv"
    cm_plot_save_name_KO = 'ConfustionMatrix_train_new.png'
    cm_data_test = plot_cm_KO(cm_data_file_KO, cm_plot_save_name_KO)
    
    # make plot performence measurements 
    # testing set here
    performence_plot_save_name_WT = "Permormance_test.png"
    model_performence(cm_data_test_WT, performence_plot_save_name_WT)
    
    # training set here 
    performence_plot_save_name_KO = "Permormance_train.png"
    model_performence(cm_data_test_KO, performence_plot_save_name_KO)
    
    # part 1.2 compare grooming prediction in WT and KO(individual and total)
    # start comparing WT and KO (or training and testting sets)
    # load predicted lables and reorganize data
    train_path = "predicted_labels/KO/"
    test_path = "predicted_labels/WT/"

    train = sorted(os.listdir(train_path))
    test = sorted(os.listdir(test_path))
    # ignore all hidden files 
    train = [f for f in train if not f.startswith('.')]
    test = [f for f in test if not f.startswith('.')]
    print(" Number of training mice:", len(train), '\n', 
          "Number of testing mice:", len(test))
    
    # reformat the loaded data
    # need total grooming time and non grooming time 
    WT_grooming_time = []
    WT_non_grooming_time = []
    WT_names = []

    for i in train:
        grooming, non_grooming = calculate_time(train_path+i)
        WT_grooming_time.append(grooming)
        WT_non_grooming_time.append(non_grooming)
        WT_names.append(i[:2])

    KO_grooming_time = []
    KO_non_grooming_time = []
    KO_names = []

    for i in test:
        grooming, non_grooming = calculate_time(test_path+i)
        KO_grooming_time.append(grooming)
        KO_non_grooming_time.append(non_grooming)
        KO_names.append(i[:2])    
    
    # make plot for grooming time and nongrooming time for each mouse(for comparing across mice and type)
    fig = plt.figure(figsize =(8,10))
    width = 0.35       # the width of the bars

    ax = plt.subplot(211)
    ind = np.arange(len(WT_grooming_time))
    rects1 = ax.bar(ind, WT_grooming_time, width, color='royalblue')
    rects2 = ax.bar(ind+width, WT_non_grooming_time, width, color='seagreen')
    ax.set_ylabel('Counts')
    ax.set_title('Train', fontsize = 20)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(WT_names)
    ax.legend( (rects1[0], rects2[0]), ('Grooming', 'Non_Grooming') )
    ax.set_ylim(0,220000)


    ax = plt.subplot(212)
    ind = np.arange(len(KO_grooming_time))
    rects1 = ax.bar(ind, KO_grooming_time, width, color='royalblue')
    rects2 = ax.bar(ind+width, KO_non_grooming_time, width, color='seagreen')
    ax.set_title('Test', fontsize = 20)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(KO_names)
    ax.legend( (rects1[0], rects2[0]), ('Grooming', 'Non_Grooming') )
    ax.set_ylim(0,220000)

    plt.tight_layout()
    fig.savefig("bar_train_test.png")
       
    
    # plot average grooming and nongrooming, then compare difference in total grooming bewteen KO and WT
    fig = plt.figure(figsize = (8,10))

    mean_groom = [np.mean(WT_total), np.mean(KO_total)]
    yerr_groom = [np.std(WT_total), np.std(KO_total)]
    mean_non = [np.mean(WT_non), np.mean(KO_non)]
    yerr_non = [np.std(WT_non), np.std(KO_non)]

    plt.subplot(211)
    b1 = plt.bar([0,1], mean_groom, 0.2,
           tick_label = ["WT", 'KO'],color='royalblue',
           yerr = yerr_groom, ecolor = 'red')
    b2 = plt.bar([0.2, 1.2], mean_non, 0.2,
                 color='seagreen',
           yerr = yerr_non, ecolor = 'red')

    plt.ylabel("Seconds")
    plt.legend( (b1[0], b2[0]), ('Grooming', 'Non_Grooming') )
    plt.title("Average Time for WT KO")


    plt.subplot(212)
    plt.boxplot((WT_total,KO_total),
                positions=[1, 1.5],
                labels=["WT","KO"], showmeans = True)
    plt.ylabel("Seconds")

    plt.tight_layout()
    fig.savefig("mean_and_boxplot_train_test.png")
    
    print(ttest(WT_total, KO_total,equal_var=False))
    
    # save total grooming time for later use
    temp = output_grooming_nongrooming([WT_total,WT_non],WT_names_ordered,'Total_time_summary_WT_21.csv')
    _ = output_grooming_nongrooming([KO_total,KO_non],KO_names_ordered,'Total_time_summary_KO_21.csv')
    
    # SECOND PART: plot individaul mouse prediction vs. hand labels
    # plot predicted grooming and hand labeled grooming side by side to see the performence
    train_hand_path = "hand_labels/KO/" # path to hand labeled data for KO
    train_hand_labels = sorted(os.listdir(train_hand_path))
    train_hand_labels = [f for f in train_hand_labels if not f.startswith('.')]
    
    for i in range(len(train)):
        hand_label = get_true_grooming(train_hand_path + train_hand_labels[i])
        pred_path = train_path + train[i]
        title = train[i][:2]
        fig, grooming = plot_grooming(hand_label, pred_path, title)
        fig_name = 'Compare_train_' + train[i][:2] +'.png'
        fig.savefig(fig_name)
    print("Blue shows predicted; orange shows hand labeled.")
    
    # for testing set(WT)
    test_hand_path = "hand_labels/WT/" # path to hand labeled data for WT
    test_hand_labels = sorted(os.listdir(test_hand_path))
    test_hand_labels = [f for f in test_hand_labels if not f.startswith('.')]
    for i in range(len(test)):
        hand_label = get_true_grooming(test_hand_path + test_hand_labels[i])
        pred_path = test_path + test[i]
        title = test[i][:2]
        fig,grooming = plot_grooming(hand_label, pred_path, title)
        fig_name = 'Compare_test_' + test[i][:2] +'.png'
        fig.savefig(fig_name)
    print("Blue shows predicted; orange shows hand labeled.")
