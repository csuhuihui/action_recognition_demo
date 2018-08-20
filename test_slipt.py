'''
Args:
input_dir(abs_path):for example,/home/zi/input_dir
output_dir(abs_path):for example,/home/zi/output_dir
Usage:
python input_dir output_dir
Attention:
input_dir and outdir_dir must be different
'''
import os
import cv2
import sys

# get RGB and flow
def slip_video(video_file,save_path):
    count = 0
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, pre_frame = cap.read()
    prvs = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
    while True:
        flag, frame = cap.read()
        if flag:
            count = count+1
            rgb = save_path+"/"+"img_%05d.jpg" %(count)
            flow_x = save_path+"/"+"flow_x_%05d.jpg"%(count)
            flow_y = save_path+"/"+"flow_y_%05d.jpg"%(count)
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            cv2.normalize(flow, flow, 0, 255, cv2.NORM_MINMAX)
            prvs = next
            cv2.imwrite(rgb,frame)
            cv2.imwrite(flow_x,flow[:,:,0])
            cv2.imwrite(flow_y, flow[:,:,1])
        else:
            break




