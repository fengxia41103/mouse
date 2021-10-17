import cv2
import os
import pandas as pd
import numpy as np

COLORS=[(255,255,255)]


def print_frames(data, connections, circle_radius=5, fn = '../video/D1_90min_black.mp4' ):
    """
    Draws pose of the mouse as found by DLC without the mouse background.
    """
    
    parts=[]
    [parts.append(key[0]) for key in data.keys() if key[0] not in parts]

    count = 0
    print("{} frames to process.".format(len(data)))
    # file opreation?
    result = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"),30.0, (640, 480), False)

    ##### really here
    #print(data.index[0], len(data)+data.index[0])
    for row in range(data.index[0], len(data)+data.index[0]):
        if count%1800==0:
            print("%f minutes processed." % (float(row)/1800))

        image = np.zeros([480,640],dtype=np.uint8)
        #image.fill(255)
        
        #print("strat", len(connections))
        for connection in connections:
            
            start, end=fetch_point(data, connection[0], row), fetch_point(data, connection[1], row)
            #print(start, end)
            cv2.line(image, start, end, (255, 255, 255), 3)

        #print("connection done", len(parts))
        for i in range(len(parts)):
            
            part=parts[i]
            pts=fetch_point(data, part, row)
            cv2.circle(image, pts, circle_radius, (255,255,255), -1)
            
        #print("points done")

        result.write(image)
        #print("write")
        count += 1
    result.release()

def fetch_point(data, part, row):
    return int(data[part]['x'][row]), int(data[part]['y'][row])

if __name__=="__main__":
    path = "/home/bizon/Desktop/single_mouse_videos/caroline_dlc_accuracy_eval/july_dlc_500_frames-caroline-2021-07-08/video_analysis/all_batches/data/"
    fn= path + "D1#-2020-09-17T20_56_41DLC_resnet50_july_dlc_500_framesJul8shuffle1_500000.h5"

    #fn = 'C1#-2020-07-18T18_21_34DLC_resnet50_july_dlc_500_framesJul8shuffle1_500000.h5'

    Data=pd.read_hdf(fn, stop=1800*90)
    scorer=Data.keys()[0][0]
    data=Data[scorer]

    connections=[('nose', 'l_ear'), ('nose', 'r_ear'), ('l_ear', 'head_base'), ('r_ear', 'head_base'),
                    ('head_base', 'c_center'), ('l_center', 'c_center'), ('r_center', 'c_center'),
                    ('c_center', 'tail_base'), ('tail_base', 'tail_tip'), ('head_base', 'f_r_shoulder'),
                    ('head_base', 'f_l_shoulder'), ('l_center', 'f_l_shoulder'), ('r_center', 'f_r_shoulder'),
                     ('l_center', 'r_l_shoulder'), ('r_center', 'r_r_shoulder'),
                    ('r_l_shoulder', 'tail_base'), ('r_r_shoulder','tail_base')]

    print_frames(data, connections)

"""
ffmpeg -r 30 -f image2 -s 640x480 -i frame_%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
"""

