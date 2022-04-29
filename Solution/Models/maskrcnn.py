from typing import Any
import numpy as np
import torch
import torchvision

# Adapted from: https://colab.research.google.com/drive/1bWLB3tmWv4XyJSu4DHtZ0-b-n-DACSKx?usp=sharing
class MaskRCNN():
    def __init__(self, device: torch.device):
        # The 91 COCO class names
        self.coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Generate a set of color for drawing different classes
        self.COLORS = np.random.uniform(0, 255, size=(len(self.coco_names), 3))

        # Initialize the model and set it to the evaluation mode
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, num_classes=91)
        self.model.to(device).eval()
        self.device = device


    def process_frames(self, frames:Any, threshold:float=0.965):
        frames_tensor = frames.to(self.device) # add a batch dimension
        with torch.no_grad():
            outputs = self.model(frames_tensor)


        frame_results = []
        for i in range(len(frames)):
            scores_filter = outputs[i]['scores'].detach()>threshold
            thresholded_objects_count = int(scores_filter.sum().detach().cpu())

            scores = outputs[i]['scores'][:thresholded_objects_count].detach().cpu().numpy()
           
            all_masks = (outputs[i]['masks']>0.5).detach().squeeze(1)
            masks = all_masks[:thresholded_objects_count].detach().cpu().numpy()
            
            boxes = outputs[i]['boxes'][:thresholded_objects_count].detach().cpu()
            boxes = [[(i[0], i[1]), (i[2], i[3])]  for i in boxes] # Get the bounding boxes, in (x1, y1), (x2, y2) format
           
            labels_raw = outputs[i]['labels'][:thresholded_objects_count].detach().cpu()
            labels = [self.coco_names[lbl_idx] for lbl_idx in labels_raw] # Get the classes labels
            frame_results.append({'boxes':boxes, 'labels': labels, 'scores':scores, 'labels_raw':labels_raw, 'masks':masks})
        return frame_results