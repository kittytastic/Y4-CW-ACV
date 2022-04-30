import sys
sys.path.append("../")
from Helpers.image_dir import ImageDirReader
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV
from Models.augmentation import movie_bg_tf, game_bg_tf, patch_tf
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import random


def show_tile(img, path):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imwrite(path, img)


def view_background():
    h,w, = 2,10
    bg_A = ImageDirReader("../../Dataset/Generated/Background/testA", transform=game_bg_tf)
    img_ids = random.sample(range(len(bg_A)), h*w)
    A_imgs = [bg_A[id] for id in img_ids]
    A_batch = torch.stack(A_imgs, dim=0)
    bg_B = ImageDirReader("../../Dataset/Generated/Background/testB", transform=movie_bg_tf)
    img_ids = random.sample(range(len(bg_A)), h*w)
    B_imgs = [bg_B[id] for id in img_ids]
    B_batch = torch.stack(B_imgs, dim=0)
    full_batch = torch.stack(A_imgs+B_imgs, dim=0)

    grid = torchvision.utils.make_grid(full_batch, nrow=w)
    show_tile(tensor_to_openCV(grid), "../../Artifacts/Q1_4_background.jpg")


def view_patch():
    h,w, = 1,10
    imgs = {}
    batches = {}
    grids = {}
    all_imgs = []
    for p in ["full-body", "full-sitting", "half-body", "head", "other"]:
        dir_read = ImageDirReader(f"../../Dataset/Generated/Poses/{p}", transform=patch_tf)
        img_ids = random.sample(range(len(dir_read)), h*w)
        imgs[p] = [dir_read[id] for id in img_ids]
        batches[p] = torch.stack(imgs[p])
        grids = torchvision.utils.make_grid(batches[p], nrow=w)
        all_imgs += imgs[p]
        #save_grid(grids, f"../../Artifacts/Q1_3_{p}.jpg")

    batch = torch.stack(all_imgs, dim=0)
    full_grid = torchvision.utils.make_grid(batch, nrow=w)

    show_tile(tensor_to_openCV(full_grid), "../../Artifacts/Q1_4_patches.jpg")


if __name__=="__main__":
    view_background()
    view_patch()