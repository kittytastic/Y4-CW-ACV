from typing import Any, List
import numpy as np
from sklearn.utils import shuffle
import torch
import torchvision
from Helpers.video import VideoReader
from Helpers.video_loader import VideoLoader
from Helpers.image_dir import IOBase
from Helpers.torch import init_torch
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.json import JsonDataLoader, JsonClassLoader
from tqdm import tqdm
import argparse
import os.path
import cv2
from Models.keypointrcnn import KeyPointRCNN        
from Models.PoseEstimator import ClassifyPose
from torchvision import transforms
import wandb




def vis_frame():
    image = cv2.imread("../Dataset/Example/PeepsShot.png")
    device = init_torch()
    keypoint_rcnn = KeyPointRCNN(device)
    frame = openCV_to_tensor(image).unsqueeze(0)

    f1_r = keypoint_rcnn.process_frames(frame)[0]
    boxes, labels, keypoints, labels_raw = f1_r['boxes'], f1_r['labels'], f1_r['keypoints'], f1_r['labels_raw']
    classify_pose(f1_r)
    for i in range(len(boxes)):
        color = keypoint_rcnn.LABEL_COLORS[labels_raw[i]]
        cv2.rectangle(image, (int(boxes[i][0][0]),int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])), color, 2) # Draw the bounding boxes
        cv2.putText(image , f"{labels[i]} {i}", (int(boxes[i][0][0]), int(boxes[i][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) # Draw the class label as text
        for j, fp in enumerate(keypoints[i]):
            cv2.circle(image, (int(fp[0]), int(fp[1])), 2, keypoint_rcnn.KP_COLORS[j], 3)

    IOBase.view_frame(image)




def classify_pose(frame_results):
    keypoint_scores, keypoints = frame_results['keypoints_score'], frame_results['keypoints']

    for i in range(len(keypoint_scores)):
        print()
        print("Person: {i}")
        for j in range(len(keypoint_scores[i])):
            print(f"{KeyPointRCNN.coco_keypoints[j]}: {keypoints[i][j]}  ({keypoint_scores[i][j]})")


if __name__=="__main__":
    print("------ Pose Estimation ------")

    dataset = JsonClassLoader("../Dataset/Extra/PoseClasses/Keypoints/")
    testset_len = len(dataset)//5
    trainset_len = len(dataset)-testset_len
    train_set, test_set = torch.utils.data.random_split(dataset, [trainset_len,testset_len])

    train_loader = torch.utils.data.DataLoader(train_set, num_workers=0, batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, num_workers=0, batch_size=1, shuffle=True)

    device = init_torch()
    print(f"Training set: {trainset_len} items")
    print(f"Test set    : {testset_len} items")


    model = ClassifyPose()
    wandb.init(project="pose-estimate-model", mode="disabled")
    wandb.watch(model, log_freq=100)
    model = model.to(device)
    model.do_training(device, 5, train_loader, test_loader)
    model.check_accuracy(device, test_loader, verbose=True)
    model.checkpoint("../Checkpoints/PoseEstimate", "final", verbose=True)