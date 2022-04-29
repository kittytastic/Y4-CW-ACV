from Helpers.image_dir import ImageDirReader
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV
from Models.augmentation import movie_bg_tf, game_bg_tf, patch_tf
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image


def show_tile(img):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imshow('Tile', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def view_game_background():
    h,w, = 5,5
    bg_A = ImageDirReader("../Dataset/Generated/Background/testA", transform=game_bg_tf)
    imgs = [bg_A[0] for _ in range(h*w)]
    batch = torch.stack(imgs, dim=0)

    grid = torchvision.utils.make_grid(batch, nrow=w)
    show_tile(tensor_to_openCV(grid))

def view_movie_background():
    h,w, = 5,5
    bg_A = ImageDirReader("../Dataset/Generated/Background/testB", transform=movie_bg_tf)
    imgs = [bg_A[i*7] for i in range(h*w)]
    batch = torch.stack(imgs, dim=0)

    grid = torchvision.utils.make_grid(batch, nrow=w)
    show_tile(tensor_to_openCV(grid))


def view_patch():
    h,w, = 5,5
    bg_A = ImageDirReader("../Dataset/Generated/HumanPatches/Games/Patches", transform=patch_tf)
    imgs = [bg_A[i] for i in range(h*w)]
    batch = torch.stack(imgs, dim=0)

    grid = torchvision.utils.make_grid(batch, nrow=w)
    show_tile(tensor_to_openCV(grid))

if __name__=="__main__":
    view_movie_background()
    view_game_background()
    view_patch()