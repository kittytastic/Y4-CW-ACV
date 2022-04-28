from collections import OrderedDict
import os
from typing import Any, Dict
import torch
import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images
from pytorch_CycleGAN_and_pix2pix.util import html

from Helpers.image_resize_loader import ImageStandardizeDataLoader, ImageStandardizer
from Helpers.ab_loader import Custom_AB_Loader, Aligned_Class_Unaligned_Data_AB_Loader
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour, custom_cgan_train, inject_time_arg
import torch
import os
from tqdm import tqdm
import argparse

from Helpers.state import StateTemplate, State




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

    st = StateTemplate()
    st.register_stage("make_frames", {"current_frame": 10, "finished": False})
    st.register_stage("get_patches", {"current_frame": 10, "finished": True})

    state = State(st, os.path.join(args.scratch_dir, "state.json"))
    print(state)

    mfs = state["make_frames"]
    print(mfs)
    print(mfs["current_frame"])
    mfs["current_frame"]+=1
    print(mfs)
    print()
    print(state)

    state.save()

    state.restore_to_default()
    print(state)



    pass
