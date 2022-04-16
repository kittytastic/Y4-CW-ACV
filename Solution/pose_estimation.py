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


class KeyPointRCNN():
    def __init__(self, device: torch.device):
        # The 91 COCO class names
        self.coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        self.coco_keypoints = [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear',
            'right_ear',
            'left_shoulder',
            'right_shoulder',
            'left_elbow',
            'right_elbow',
            'left_wrist',
            'right_wrist',
            'left_hip',
            'right_hip',
            'left_knee',
            'right_knee',
            'left_ankle',
            'right_ankle'
        ]

        # Generate a set of color for drawing different classes
        self.LABEL_COLORS = np.random.uniform(0, 255, size=(len(self.coco_names), 3))
        self.KP_COLORS = np.random.uniform(0, 255, size=(len(self.coco_keypoints), 3))

        # Initialize the model and set it to the evaluation mode
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, progress=True)
        self.model.to(device).eval()
        self.device = device


    def process_frames(self, frames:Any, threshold:float=0.965):
        frames_tensor = frames.to(self.device) # add a batch dimension
        with torch.no_grad():
            outputs = self.model(frames_tensor)

        for k,v in outputs[0].items():
            print(f"{k}:  {v.shape}")

        frame_results = []
        for i in range(len(frames)):
            scores_filter = outputs[i]['scores'].detach()>threshold
            thresholded_objects_count = int(scores_filter.sum().detach().cpu())

            scores = outputs[i]['scores'][:thresholded_objects_count].detach().cpu().numpy()
            
            boxes = outputs[i]['boxes'][:thresholded_objects_count].detach().cpu()
            boxes = [[(i[0], i[1]), (i[2], i[3])]  for i in boxes] # Get the bounding boxes, in (x1, y1), (x2, y2) format
           
            labels_raw = outputs[i]['labels'][:thresholded_objects_count].detach().cpu()
            labels = [self.coco_names[lbl_idx] for lbl_idx in labels_raw] # Get the classes labels
            
            keypoints = outputs[i]['keypoints'][:thresholded_objects_count].detach().cpu().numpy()
            keypoint_scores = outputs[i]['keypoints_scores'][:thresholded_objects_count].detach().cpu().numpy()
            frame_results.append({'boxes':boxes, 'labels': labels, 'keypoints':keypoints, 'keypoints_score':keypoint_scores, 'scores':scores, 'labels_raw':labels_raw})
        return frame_results
        

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