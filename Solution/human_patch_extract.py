from typing import Any, List
import numpy as np
import torch
import torchvision
from Helpers.video import VideoReader
from Helpers.video_loader import VideoLoader
from Helpers.image_dir import ImageDirWriter
from Helpers.torch import init_torch
from tqdm import tqdm
import argparse
import os.path

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
            scores_tensor = outputs[i]['scores'].detach()
            scores = scores_tensor>threshold
            thresholded_objects_count = int(scores.sum().detach().cpu())
           
            all_masks = (outputs[i]['masks']>0.5).detach().squeeze()
            masks = all_masks[:thresholded_objects_count].detach().cpu().numpy()
            
            all_boxes = outputs[i]['boxes'].detach()
            boxes = all_boxes[:thresholded_objects_count].cpu()
            boxes = [[(i[0], i[1]), (i[2], i[3])]  for i in boxes] # Get the bounding boxes, in (x1, y1), (x2, y2) format
           
            all_labels = outputs[i]['labels'].detach()
            labels = all_labels[:thresholded_objects_count].detach().cpu()
            labels = [self.coco_names[lbl_idx] for lbl_idx in labels] # Get the classes labels
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



def full_mode(workers:int, batch_size: int):
    device = init_torch()
    mask_rcnn = MaskRCNN(device)

    DATA_BASE_PATH = "../Dataset/Train"
    game_videos = ["Video1.mp4", "Video2.mp4"]
    movie_videos = ["Video1.mp4", "Video2.mp4", "Video3.mp4", "Video4.mp4", "Video5.mp4", "Video6.mp4", "Video7.mp4", "Video8.mp4", "Video9.mp4"]


    game_outstream = ImageDirWriter("../Dataset/Generated/HumanPatches/Games")
    movie_outstream = ImageDirWriter("../Dataset/Generated/HumanPatches/Movie")
    
    total_samples = 0
    print(f"Processing {len(game_videos)} game videos")
    for idx, game in enumerate(game_videos):
        print(f"\tStarting: {game}   ({idx+1})")
        video_path = os.path.join(DATA_BASE_PATH, "Games", game)
        video = VideoReader(video_path)
        ds = VideoLoader(video_path)
        data = torch.utils.data.DataLoader(ds, num_workers=workers, batch_size=batch_size)
        samples = 0
        
        for tensor, opencv  in tqdm(data, total = len(video)//batch_size):
            people = mask_rcnn.get_humans(tensor, opencv)
            samples += len(people)
            game_outstream.write_frames(people)
        
        print(f"\t\t{samples} samples")
        total_samples += samples
    print(f"Total samples: {total_samples}")
    print()
    total_samples = 0
    print(f"Processing {len(movie_videos)} movie videos")
    for idx, movie in enumerate(movie_videos):
        print(f"\tStarting: {movie}   ({idx+1})")
        video_path = os.path.join(DATA_BASE_PATH, "Movie", movie)
        video = VideoReader(video_path)
        ds = VideoLoader(video_path)
        data = torch.utils.data.DataLoader(ds, num_workers=workers, batch_size=batch_size)
        samples = 0
        
        for tensor, opencv  in tqdm(data, total = len(video)//batch_size):
            people = mask_rcnn.get_humans(tensor, opencv)
            samples += len(people)
            movie_outstream.write_frames(people)
        
        print(f"\t\t{samples} samples")
        total_samples += samples
    print(f"Total samples: {total_samples}")

if __name__=="__main__":
    print("------ Human Patch Extract ------")
    
    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-b', '--batch', type=int,help='Batch Size', default=16)
    parser.add_argument('-w', '--workers', type=int, help='Number of Workers', default=2)
    parser.add_argument('-m', '--mode', type=str, help='Operation Mode', default="full")
    args = parser.parse_args()

    batch_size = args.batch
    workers = args.workers

    if args.mode == "full":
        full_mode(workers, batch_size)
    else:
        print(f"Unknown mode: {args.mode}")
    