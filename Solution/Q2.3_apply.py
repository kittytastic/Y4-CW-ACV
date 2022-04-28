from collections import OrderedDict
import os
from typing import Any, Dict
import torch
import sys
from Helpers.torch import init_torch
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images
from pytorch_CycleGAN_and_pix2pix.util import html

from Helpers.image_resize_loader import ImageStandardizeDataLoader, ImageStandardizer
from Helpers.video import VideoReader
from Helpers.ab_loader import Custom_AB_Loader, Aligned_Class_Unaligned_Data_AB_Loader
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour, custom_cgan_train, inject_time_arg
from Models.keypointrcnn import KeyPointRCNN
from Models.maskrcnn import MaskRCNN
import torch
import os
from tqdm import tqdm, trange
import argparse
import cv2
import json
import numpy as np

from Helpers.state import StateTemplate, State, StageState

def turn_video_to_frames(state: StageState, video_path: str, scratch_dir: str):
    if state["finished"]: return
    current_frame = state["current_frame"]

    out_directory = os.path.join(scratch_dir, "raw_frames")
    video_reader = VideoReader(video_path)
    video_reader.seek(current_frame)
    total_frames = len(video_reader)

    pbar = tqdm(video_reader, total=total_frames, initial=current_frame)
    pbar.set_description("Saving Frames")
    for f in pbar:
        cv2.imwrite(os.path.join(out_directory, f"frame-{current_frame}.jpg"), f)
        if current_frame % 30==0: state["current_frame"] = current_frame 
        current_frame += 1

    state["finished"]=True


def get_patches(state: StageState, props:Dict[str, Any], device: Any, scratch_dir: str):
    if state["finished"]: return
    current_frame = state["current_frame"]
    total_frames = props["total_frames"]
    in_dir = os.path.join(scratch_dir, "raw_frames")
    out_dir = os.path.join(scratch_dir, "keypoint_rcnn")

    model = KeyPointRCNN(device)

    pbar = trange(current_frame, total_frames)
    pbar.set_description("Running Keypoint RCNN")
    for current_frame in pbar:
        frame_cv = cv2.imread(os.path.join(in_dir, f"frame-{current_frame}.jpg"))
        frame_tensor = openCV_to_tensor(frame_cv).unsqueeze(0)
        frame_results = model.process_frames(frame_tensor)[0]

        boxes, labels, key_points, key_points_scores, scores = frame_results["boxes"], frame_results["labels"], frame_results["keypoints"], frame_results["keypoints_score"], frame_results["scores"]
        
        entity_count = 0
        entities_data = []
        for i in range(len(labels)):
            if labels[i]!="person": continue
            
            x1, y1, x2, y2 = int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]), int(boxes[i][1][1])
            
            entity_data = {
                "box": [x1, y1, x2, y2],
                "score": float(scores[i]),
                "keypoints": key_points[i].tolist(),
                "keypoint_scores": key_points_scores[i].tolist(),
                }
            entities_data.append(entity_data)
            
            person_image = frame_cv[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(out_dir, "patch", f"frame-{current_frame}-entity-{entity_count}.jpg"), person_image)
            entity_count+=1
        
        with open(os.path.join(out_dir, "data", f"frame-{current_frame}.json"), "w+") as f: json.dump(entities_data, f)

        if current_frame%30==0: state["current_frame"]=current_frame

    state["finished"]=True


def get_masks(state: StageState, props:Dict[str, Any], device: Any, scratch_dir: str):
    if state["finished"]: return
    current_frame = state["current_frame"]
    total_frames = props["total_frames"]
    in_dir = os.path.join(scratch_dir, "raw_frames")
    out_dir = os.path.join(scratch_dir, "mask_rcnn")

    model = MaskRCNN(device)


    frame_cv = cv2.imread(os.path.join(in_dir, f"frame-{current_frame}.jpg"))
    frame_tensor = openCV_to_tensor(frame_cv).unsqueeze(0)
    frame_results = model.process_frames(frame_tensor)[0]

    
    boxes, labels, masks, scores = frame_results["boxes"], frame_results["labels"], frame_results["masks"], frame_results["scores"]
    
    
    pbar = trange(current_frame, total_frames)
    pbar.set_description("Running Keypoint RCNN")
    for current_frame in pbar:
        entity_count = 0
        entities_data = []
        for i in range(len(labels)):
            if labels[i]!="person": continue
            
            x1, y1, x2, y2 = int(boxes[i][0][0]), int(boxes[i][0][1]), int(boxes[i][1][0]), int(boxes[i][1][1])
            
            entity_data = {
                "box": [x1, y1, x2, y2],
                "score": float(scores[i]),
                }
            entities_data.append(entity_data)

            if props["debug_mask_rcnn"]:    
                person_image = frame_cv[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(out_dir, "patch", f"frame-{current_frame}-entity-{entity_count}.jpg"), person_image)

            mask_bw = np.full(masks[i].shape, 255, dtype=np.uint8)
            mask_bw *= masks[i]
            cv2.imwrite(os.path.join(out_dir, "masks", f"frame-{current_frame}-entity-{entity_count}.png"), mask_bw)
            
            entity_count+=1
        
        with open(os.path.join(out_dir, "data", f"frame-{current_frame}.json"), "w+") as f: json.dump(entities_data, f)
    
    state["finished"]=True
    





def make_props(video_path: str):
    video_reader = VideoReader(video_path)
    total_frames = len(video_reader)

    return {"total_frames":total_frames, "debug_mask_rcnn": True}

if __name__=="__main__":
    # Find Patches
    # Find Masks
    # Match patches
    # Apply background model
    
    # Human patches
    # -> Apply patch model
    # -> Apply patches to background

    # VR -> Frames
    # Per frame:
    #   background-{frame}.jpg
    #   frame-info-{frame}.jpg (entities, keypoints, patch start, mask start)
    #   human-{frame}-{entity}.jpg
    #   mask-{frame}-{entity}.jpg
    print("------ Q2.3 - Apply to test video ------")
    
    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-i', '--input', type=str,help='Input video', default="../Dataset/Test/Video1.mp4")
    parser.add_argument('-s', '--scratch_dir', type=str, help='Scratch Directory', default="../Dataset/Test/Video1")
    args = parser.parse_args()

    assert(os.path.isfile(args.input))
    assert(os.path.isdir(args.scratch_dir))

    device = init_torch()

    st = StateTemplate()
    st.register_stage("make_frames", {"current_frame": 10, "finished": False})
    st.register_stage("find_patches", {"current_frame": 10, "finished": True})
    st.register_stage("find_masks", {"current_frame": 10, "finished": True})

    state = State(st, os.path.join(args.scratch_dir, "state.json"))
    props = make_props(args.input)
  
    turn_video_to_frames(state["make_frames"], args.input, args.scratch_dir)
    state.save(pretty_print=True)
    print("Turn video to frames:  ✔️")

    get_patches(state["find_patches"], props, device, args.scratch_dir)
    state.save(pretty_print=True)
    print("Run keypoint RCNN:  ✔️     (gets patches, bounding boxes, keypoints)")
    
    get_masks(state["find_masks"], props, device, args.scratch_dir)
    state.save(pretty_print=True)
    print("Run keypoint RCNN:  ✔️     (gets patches, bounding boxes, masks)")



    pass
