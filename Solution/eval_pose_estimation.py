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
        dir_path = os.path.join(outpath, c)
        if not os.path.exists(dir_path): os.makedirs(dir_path)


def process_directory(device, model, source_patten, data_patten, target_patten, data_dir, target_dir):
    dataset = JsonDataLoader(os.path.join(data_dir, "Keypoints"))
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)

    model.eval()
    with torch.no_grad(): 
        source_patten = parse.compile(source_patten) 
        for data_raw in tqdm(data_loader):
            data_tensor = model.pack_keypoint_tensor(data_raw)
            labels = model.classify(device, data_tensor).detach().cpu().numpy()
            for i in range(len(data_tensor)):
                #print(f"{data_raw['file_name'][i]}  is {labels[i]}")
                match = source_patten.parse(data_raw['file_name'][i])
                assert(match is not None)
                class_name = classes[labels[i]]
                data_name = data_patten.format(**match.named)
                target_name = target_patten.format(**match.named)
                data_path = os.path.join(data_dir, "Patches", data_name)
                target_path = os.path.join(target_dir, class_name, target_name)
                shutil.copyfile(data_path, target_path)

if __name__=="__main__":
    print("------ Pose Estimation - Eval ------")
    parser = argparse.ArgumentParser(description='Pose Estimation Eval')
    parser.add_argument('-o', '--outdir', type=str, help='Output Directory', default="../Dataset/Generated/Poses")
    args = parser.parse_args()
    assert(os.path.isdir(args.outdir))

    device = init_torch()

    model = ClassifyPose()
    classes = model.restore("../Checkpoints/PoseEstimate", "final")
    make_dirs(args.outdir, classes)
    model = model.to(device)

    print("\nProcessing Games")
    process_directory(device, model, "output-{id}.json", "output-{id}.png", "game-output-{id}.png", "../Dataset/Generated/HumanPatches/Games", args.outdir) 
    print("Processing Movies")
    process_directory(device, model, "output-{id}.json", "output-{id}.png", "movie-output-{id}.png", "../Dataset/Generated/HumanPatches/Movie", args.outdir)
        