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
    h,w, = 5,10
    bg_A = ImageDirReader("../../Dataset/Generated/HumanPatches/Games/Patches", transform=fix_shape)
    img_ids = random.sample(range(len(bg_A)), h*w)
    A_imgs = [bg_A[id] for id in img_ids]
    bg_B = ImageDirReader("../../Dataset/Generated/HumanPatches/Movie/Patches", transform=fix_shape)
    img_ids = random.sample(range(len(bg_B)), h*w)
    B_imgs = [bg_B[id] for id in img_ids]
    A_batch = torch.stack(A_imgs)
    B_batch = torch.stack(B_imgs)
    batch = torch.stack(A_imgs[:w*(h-2)]+B_imgs[:w*(h-2)], dim=0)

    A_grid = torchvision.utils.make_grid(A_batch, nrow=w)
    B_grid = torchvision.utils.make_grid(B_batch, nrow=w)
    full_grid = torchvision.utils.make_grid(batch, nrow=w)

    save_grid(A_grid, "../../Artifacts/Q1_1_A.jpg")
    save_grid(B_grid, "../../Artifacts/Q1_1_B.jpg")
    save_grid(full_grid, "../../Artifacts/Q1_1_full.jpg")

if __name__=="__main__":
    gen_figure()