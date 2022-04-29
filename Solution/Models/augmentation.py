from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV
from Helpers.image_resize_loader import ImageStandardizer
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image

_resizer = ImageStandardizer(256, 256)
def patch_tf(img):
    img = openCV_to_PIL(img)
    h,w = __make_power_2(img, 4)
    transforms = torch.nn.Sequential(
        torchvision.transforms.ColorJitter(brightness=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomPerspective(distortion_scale=0.5),
        #torchvision.transforms.Resize((256, 256)),
    )
    img = transforms(img)
    img, size = _resizer.standardize_size(np.array(img))
    img = PIL_to_tensor(img)
    return img


def game_bg_tf(img):
    img = openCV_to_PIL(img)
    h,w = __make_power_2(img, 4)
    transforms = torch.nn.Sequential(
        torchvision.transforms.ColorJitter(brightness=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomPerspective(distortion_scale=0.5),
        torchvision.transforms.RandomResizedCrop((256,256), scale=(0.08, 1.0)), 
    )
    img = transforms(img)
    img = PIL_to_tensor(img)
    return img

def movie_bg_tf(img):
    return game_bg_tf(img)

# from pytorch_CycleGAN_and_pix2pix/data/base_dataset.py
def __make_power_2(img, base):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    return h,w