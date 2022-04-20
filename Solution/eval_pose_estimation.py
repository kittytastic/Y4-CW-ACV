from typing import Any, List
import numpy as np
import torch
from Helpers.torch import init_torch
from Helpers.json import JsonDataLoader
from tqdm import tqdm
import argparse
import os
from Models.PoseEstimator import ClassifyPose
import parse
import shutil


def make_dirs(outpath, classes):
    for c in classes.values():
        os.makedirs(os.path.join(outpath, c))

if __name__=="__main__":
    print("------ Pose Estimation - Eval ------")
    parser = argparse.ArgumentParser(description='Extract Human Patches')
    parser.add_argument('-o', '--outdir', type=str, help='Input Directory', default="../Dataset/Generated/Poses")
    parser.add_argument('-i', '--indir', type=str, help='Output Directory', default="./todo")
    parser.add_argument('-s', '--source', type=str, help='Source Name', default="output-{id}.json")
    parser.add_argument('-d', '--data', type=str, help='Data Name', default="output-{id}.png")
    parser.add_argument('-t', '--target', type=str, help='Target Name', default="output-{id}.png")
    args = parser.parse_args()
    assert(os.path.isdir(args.outdir))
    assert(os.path.isdir(args.indir))

    dataset = JsonDataLoader("../Dataset/Generated/HumanPatches/Games/Keypoints/")

    test_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)

    device = init_torch()


    model = ClassifyPose()
    classes = model.restore("../Checkpoints/PoseEstimate", "final")
    make_dirs(args.outdir, classes)
    model = model.to(device)

    data_raw = next(iter(test_loader))
    data_tensor = model.pack_keypoint_tensor(data_raw)
    labels = model.classify(device, data_tensor).detach().cpu().numpy()



    source_patten = parse.compile(args.source) 
    for i in range(len(data_tensor)):
        print(f"{data_raw['file_name'][i]}  is {labels[i]}")
        match = source_patten.parse(data_raw['file_name'][i])
        assert(match is not None)
        class_name = classes[labels[i]]
        data_name = args.data.format(**match.named)
        target_name = args.target.format(**match.named)
        data_path = os.path.join(args.indir, data_name)
        target_path = os.path.join(args.outdir, class_name, target_name)
        print(data_path)
        print(target_path)
        shutil.copyfile(data_path, target_path)
        
    print(labels) 