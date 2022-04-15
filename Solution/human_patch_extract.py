from typing import Any
import cv2
import numpy as np
import random
import torch
import torchvision
from PIL import Image
from IPython import display
from torchvision.transforms import transforms
from Helpers.video import VideoReader

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


    def process_frame(self, frame: np.ndarray, threshold:float=0.965):
        # Transform to convert the image to tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Analyze the image
        in_frame_processing = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert from CV2's BGR to RGB
        in_frame_processing = transform(in_frame_processing) # Convert the image to tensor
        in_frame_processing = in_frame_processing.unsqueeze(0).to(device) # add a batch dimension
        with torch.no_grad():
            outputs = self.model(in_frame_processing)

        # Get individual types of output from the outputs variable
        scores = list(outputs[0]['scores'].detach().cpu().numpy()) # Get scores
        thresholded_objects = [scores.index(i) for i in scores if i > 0.965] # Get an index for the objects having the scores > a threshold of 0.965
        thresholded_objects_count = len(thresholded_objects) # Total objects having scores > threshold
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy() # Get the segmentation masks
        masks = masks[:thresholded_objects_count] # Discard masks for objects that are below threshold by only taking the beginning of the list
        boxes = [[(i[0], i[1]), (i[2], i[3])]  for i in outputs[0]['boxes'].detach().cpu()] # Get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = boxes[:thresholded_objects_count] # Discard bounding boxes for objects that are below threshold by only taking the beginning of the list
        labels = [self.coco_names[i] for i in outputs[0]['labels']] # Get the classes labels
        return scores, masks, boxes, labels
    
    def get_humans(self, frame:np.ndarray):
        scores, masks, boxes, labels = self.process_frame(frame)

        # Get the image 
        out_frame = frame.copy()
        out_frame = np.array(out_frame)

        # Draw the segmentation masks with the text labels
        for i in range(len(masks)): # For all detected objects with score > threshold
            x1, y1, x2, y2 = int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]), int(boxes[i][1][1])
            print(f"{labels[i]} : {float(scores[i])} ({x1}, {y1}, {x2}, {y2})")
            color = self.COLORS[random.randrange(0, len(self.COLORS))] # Pick a random color
            red_map = np.zeros_like(masks[i]).astype(np.uint8) # Initialize an empty mask for each of the RGB channels
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color # Set the color of the masked pixels
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2) # Combine the three channels of the mask
            cv2.addWeighted(out_frame, 1.0, segmentation_map, 0.6, 0.0, out_frame) # Apply the mask onto the image
            cv2.rectangle(out_frame, (int(boxes[i][0][0]),int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])), color, 2) # Draw the bounding boxes
            cv2.putText(out_frame , labels[i], (int(boxes[i][0][0]), int(boxes[i][0][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA) # Draw the class label as text

            person = frame[y1:y2, x1:x2]
            cv2.imwrite(f"./person{i}.png", person)

        # Save the image
        cv2.imwrite("./output.png", out_frame)


def init_torch()->torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>0:
        print(f"Using: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using: {device}")
    
    return device

if __name__=="__main__":
    print("------ Human Patch Extract ------")
    device = init_torch()
    mask_rcnn = MaskRCNN(device)
    video = VideoReader("../Dataset/Train/Games/Video1.mp4")
    frame = next(video)
    mask_rcnn.get_humans(frame)