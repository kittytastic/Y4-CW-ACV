from typing import Any, List
import cv2
import numpy as np
import random
import torch
import torchvision
from PIL import Image
from IPython import display
from torchvision.transforms import transforms
from Helpers.video import VideoReader
from Helpers.video_loader import VideoLoader
from Helpers.image_dir import ImageDirWriter
from Helpers.images import tensor_to_openCV
from tqdm import tqdm

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


    def process_frames(self, frames:Any, threshold:float=0.965):
        frames_tensor = frames.to(device) # add a batch dimension
        with torch.no_grad():
            outputs = self.model(frames_tensor)


        frame_results = []
        for i in range(len(frames)):
            # Get individual types of output from the outputs variable
            scores = list(outputs[i]['scores'].detach().cpu().numpy()) # Get scores
            thresholded_objects = [scores.index(i) for i in scores if i > 0.965] # Get an index for the objects having the scores > a threshold of 0.965
            thresholded_objects_count = len(thresholded_objects) # Total objects having scores > threshold
            masks = (outputs[i]['masks']>0.5).squeeze().detach().cpu().numpy() # Get the segmentation masks
            masks = masks[:thresholded_objects_count] # Discard masks for objects that are below threshold by only taking the beginning of the list
            boxes = [[(i[0], i[1]), (i[2], i[3])]  for i in outputs[0]['boxes'].detach().cpu()] # Get the bounding boxes, in (x1, y1), (x2, y2) format
            boxes = boxes[:thresholded_objects_count] # Discard bounding boxes for objects that are below threshold by only taking the beginning of the list
            labels = [self.coco_names[i] for i in outputs[0]['labels']] # Get the classes labels
            frame_results.append((scores, masks, boxes, labels))
        return frame_results
    
    def get_humans(self, frames_tensor:Any, frames_opencv):
        frame_results = self.process_frames(frames_tensor)
        people:List[np.ndarray] = []
        
        for idx, (scores, masks, boxes, labels) in enumerate(frame_results):
            frame = frames_opencv[idx].numpy()

            # Draw the segmentation masks with the text labels
            for i in range(len(masks)): # For all detected objects with score > threshold
                if labels[i]!="person":
                    #print(f"Skipping: {labels[i]}")
                    continue
                x1, y1, x2, y2 = int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]), int(boxes[i][1][1])
                person = frame[y1:y2, x1:x2]
                people.append(person)

        return people


def init_torch()->torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>0:
        print(f"Using: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using: {device}")
    
    return device

if __name__=="__main__":
    print("------ Human Patch Extract ------")
    batch_size = 16
    device = init_torch()
    mask_rcnn = MaskRCNN(device)
    video = VideoReader("../Dataset/Train/Games/Video1.mp4")
    outstream = ImageDirWriter("../Dataset/Generated/HumanPatches")
    ds = VideoLoader('../Dataset/Train/Games/Video1.mp4')
    data = torch.utils.data.DataLoader(ds, num_workers=4, batch_size=batch_size)
    

    for tensor, opencv  in tqdm(data, total = len(video)//batch_size):
        people = mask_rcnn.get_humans(tensor, opencv)
        outstream.write_frames(people)