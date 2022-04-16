from typing import Any, List
import numpy as np
import torch
import torchvision
from Helpers.video import VideoReader
from Helpers.video_loader import VideoLoader
from Helpers.image_dir import IOBase
from Helpers.torch import init_torch
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from tqdm import tqdm
import argparse
import os.path
import cv2
from Models.keypointrcnn import KeyPointRCNN        

def vis_frame():
    image = cv2.imread("../Dataset/Example/PeepsShot.png")
    device = init_torch()
    keypoint_rcnn = KeyPointRCNN(device)
    frame = openCV_to_tensor(image).unsqueeze(0)

    f1_r = keypoint_rcnn.process_frames(frame)[0]
    boxes, labels, keypoints, labels_raw = f1_r['boxes'], f1_r['labels'], f1_r['keypoints'], f1_r['labels_raw']
    for i in range(len(boxes)):
        color = keypoint_rcnn.LABEL_COLORS[labels_raw[i]]
        cv2.rectangle(image, (int(boxes[i][0][0]),int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])), color, 2) # Draw the bounding boxes
        cv2.putText(image , labels[i], (int(boxes[i][0][0]), int(boxes[i][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) # Draw the class label as text
        for j, fp in enumerate(keypoints[i]):
            cv2.circle(image, (int(fp[0]), int(fp[1])), 2, keypoint_rcnn.KP_COLORS[j], 3)

    IOBase.view_frame(image)

if __name__=="__main__":
    print("------ Pose Estimation ------")
    vis_frame()