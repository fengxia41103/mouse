import os
from draw_pose import *

path = "/home/bizon/Desktop/single_mouse_videos/caroline_dlc_accuracy_eval/CRNN/more_videos_here/more_WT_here/WT_DLC_outputs/"
all_data = os.listdir(path)

print("Start generating videos!")
for i in all_data:
    fn = path + i
    
    Data=pd.read_hdf(fn, stop=1800*100)
    scorer=Data.keys()[0][0]
    data=Data[scorer]

    connections=[('nose', 'l_ear'), ('nose', 'r_ear'), ('l_ear', 'head_base'), ('r_ear', 'head_base'),
                        ('head_base', 'c_center'), ('l_center', 'c_center'), ('r_center', 'c_center'),
                        ('c_center', 'tail_base'), ('tail_base', 'tail_tip'), ('head_base', 'f_r_shoulder'),
                        ('head_base', 'f_l_shoulder'), ('l_center', 'f_l_shoulder'), ('r_center', 'f_r_shoulder'),
                         ('l_center', 'r_l_shoulder'), ('r_center', 'r_r_shoulder'),
                        ('r_l_shoulder', 'tail_base'), ('r_r_shoulder','tail_base')]
    
    output = '../video/aug_13_'+ i[:15] +'_100min_black.mp4'
    print_frames(data, connections, fn = output )
    print(output)
    print("Saved!")
    
print("ALL DONE!")