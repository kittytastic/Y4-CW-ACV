from typing import Any, Dict, List, Tuple
from cv2 import normalize
import numpy as np
import torch
from Helpers.video import VideoReader
from Helpers.video_loader import VideoLoader
from Helpers.image_dir import ImageDirWriter
from Helpers.json import JsonDirWriter
from Helpers.torch import init_torch
from tqdm import tqdm
import argparse
import os.path
from Models.maskrcnn import MaskRCNN
from Models.keypointrcnn import KeyPointRCNN

    
def get_humans(model, frames_tensor:Any, frames_opencv:Any):
    frame_results = model.process_frames(frames_tensor)
    people_and_kp:List[Tuple[np.ndarray, Dict[Any, Any]]] = []
    
    for idx, frame_info in enumerate(frame_results):
        boxes, labels, key_points, key_points_scores = frame_info["boxes"], frame_info["labels"], frame_info["keypoints"], frame_info["keypoints_score"]
        frame = frames_opencv[idx].numpy()

        # Draw the segmentation masks with the text labels
        for i in range(len(boxes)): # For all detected objects with score > threshold
            if labels[i]!="person":
                #print(f"Skipping: {labels[i]}")
                continue
            x1, y1, x2, y2 = int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]), int(boxes[i][1][1])
            #print(f"({x1} -> {x2}), ({y1} -> {y2})")
            person_image = frame[y1:y2, x1:x2]
            normalised_kp = key_points[i].copy()
            normalised_kp[:,0] = (normalised_kp[:,0]-x1)/(x2-x1)
            normalised_kp[:,1] = (normalised_kp[:,1]-y1)/(y2-y1)
            person_kp = {
                "keypoints": key_points[i].tolist(),
                "nomalised_keypoints": normalised_kp.tolist(),
                "scores": key_points_scores[i].tolist()
                }
            people_and_kp.append((person_image, person_kp))

    return people_and_kp


def process_full_video(video_path:str, image_outstream:ImageDirWriter, keypoint_outstream: JsonDirWriter):
    video_info = VideoReader(video_path)
    ds = VideoLoader(video_path)
    data = torch.utils.data.DataLoader(ds, num_workers=workers, batch_size=batch_size) # type: ignore
    samples = 0
        
    for tensor, opencv  in tqdm(data, total = len(video_info)//batch_size):
        all_people = get_humans(model, tensor, opencv)
        imgs, kps = zip(*all_people)
        samples += len(all_people)
        image_outstream.write_frames(imgs)
        keypoint_outstream.write_objects(kps)
    
    return samples


def full_mode(workers:int, batch_size: int, model:Any):
    

    DATA_BASE_PATH = "../Dataset/Train"
    game_videos = ["Video1.mp4", "Video2.mp4"]
    movie_videos = ["Video1.mp4", "Video2.mp4", "Video3.mp4", "Video4.mp4", "Video5.mp4", "Video6.mp4", "Video7.mp4", "Video8.mp4", "Video9.mp4"]

    game_outstream = ImageDirWriter("../Dataset/Generated/HumanPatches/Games/Patches")
    movie_outstream = ImageDirWriter("../Dataset/Generated/HumanPatches/Movie/Patches")
    
    game_json_outstream = JsonDirWriter("../Dataset/Generated/HumanPatches/Games/Keypoints")
    movie_json_outstream = JsonDirWriter("../Dataset/Generated/HumanPatches/Movie/Keypoints")
    
    total_samples = 0
    print(f"Processing {len(game_videos)} game videos")
    for idx, game in enumerate(game_videos):
        print(f"\tStarting: {game}   ({idx+1})")
        video_path = os.path.join(DATA_BASE_PATH, "Games", game)
        samples = process_full_video(video_path, game_outstream, game_json_outstream)
        print(f"\t\t{samples} samples")
        total_samples += samples
    print(f"Total samples: {total_samples}")
    print()

    total_samples = 0
    print(f"Processing {len(movie_videos)} movie videos")
    for idx, movie in enumerate(movie_videos):
        print(f"\tStarting: {movie}   ({idx+1})")
        video_path = os.path.join(DATA_BASE_PATH, "Movie", movie)
        samples = process_full_video(video_path, movie_outstream, movie_json_outstream) 
        print(f"\t\t{samples} samples")
        total_samples += samples
    print(f"Total samples: {total_samples}")

if __name__=="__main__":
    print("------ Human Patch Extract ------")
    
    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-b', '--batch', type=int,help='Batch Size', default=12)
    parser.add_argument('-w', '--workers', type=int, help='Number of Workers', default=2)
    parser.add_argument('-m', '--mode', type=str, help='Operation Mode', default="full")
    parser.add_argument('-n', '--network', type=str, help='Network', default="kp")
    args = parser.parse_args()

    batch_size = args.batch
    workers = args.workers
    
    device = init_torch()
    if args.network == "mask":
        model = MaskRCNN(device)
    elif args.network == "kp":
        model = KeyPointRCNN(device)
    else:
        print(f"Unknown network: {args.network}")
        exit()

    
    if args.mode == "full":
        full_mode(workers, batch_size, model)
    else:
        print(f"Unknown mode: {args.mode}")
    