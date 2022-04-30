import sys
sys.path.append("../")
from Helpers.image_dir import ImageDirReader
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV, openCV_to_tensor
from Helpers.image_resize_loader import ImageStandardizer
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import random


def show_tile(img):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imshow('Tile', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_grid(img, file:str):
    img = tensor_to_openCV(img)
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imwrite(file, img)



def fix_shape(img):
    img_s = ImageStandardizer(256, 256)
    img, size = img_s.standardize_size(img)
    return openCV_to_tensor(img)



def gen_figure():
    h,w, = 1,10
    imgs = {}
    batches = {}
    grids = {}
    all_imgs = []
    for p in ["full-body", "full-sitting", "half-body", "head", "other"]:
        dir_read = ImageDirReader(f"../../Dataset/Generated/Poses/{p}", transform=fix_shape)
        img_ids = random.sample(range(len(dir_read)), h*w)
        imgs[p] = [dir_read[id] for id in img_ids]
        batches[p] = torch.stack(imgs[p])
        grids = torchvision.utils.make_grid(batches[p], nrow=w)
        all_imgs += imgs[p]
        save_grid(grids, f"../../Artifacts/Q1_3_{p}.jpg")

    batch = torch.stack(all_imgs, dim=0)
    full_grid = torchvision.utils.make_grid(batch, nrow=w)

    save_grid(full_grid, "../../Artifacts/Q1_3_full.jpg")

if __name__=="__main__":
    gen_figure()