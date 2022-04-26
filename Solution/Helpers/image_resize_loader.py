from ast import Call
from typing import Dict, Callable, Any, Optional, Tuple
import torch
import os
import numpy as np
import cv2
import math

from .image_dir import ImageDirReader

class ImageStandardizeDataLoader(torch.utils.data.Dataset): #type: ignore
    def __init__(self, data_dir:str, target_height:int, target_width:int, file_type:str="jpg", post_resize_transfrom: Optional[Callable[[Any], Any]]=None, resize_mode="pad"):
        assert(os.path.isdir(data_dir))
        super().__init__()
        self.data_reader = ImageDirReader(data_dir, file_type=file_type)
        self.post_resize_transform = post_resize_transfrom
        assert(resize_mode in {"pad", "stretch"})
        self.resize_mode = resize_mode
        self.target_height = target_height
        self.target_width = target_width

    def standardize_pad(self, img):
        img_h, img_w, _ = img.shape
        resize_img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        height_sf = self.target_height/img_h
        width_sf = self.target_width/img_w
        sf = min(height_sf, width_sf)
        new_height, new_width = int(math.floor(img_h*sf)), int(math.floor(img_w)*sf)
        og_image = cv2.resize(img, (new_width, new_height))
        resize_img[0:new_height, 0:new_width, :] = og_image
        return resize_img

    def standardize_stretch(self, img):
        img = cv2.resize(img, (self.target_width, self.target_height))
        return img
    
    def standardize_size(self, img):
        original_size = (img.shape[0], img.shape[1]) # height, width
        if self.resize_mode == "pad": resize_img = self.standardize_pad(img)
        elif self.resize_mode == "stretch": resize_img = self.standardize_stretch(img)
        else: raise Exception(f"Unknown option: {self.resize_mode}")
        return resize_img, original_size

    def restore_pad(self, img, size):
        height_sf = self.target_height/size[0]
        width_sf = self.target_width/size[1]
        sf = min(height_sf, width_sf)
        new_height, new_width = int(math.floor(size[0]*sf)), int(math.floor(size[1])*sf)
        image_pix = img[0:new_height, 0:new_width, :]
        og_image = cv2.resize(image_pix, (size[1], size[0]))
        return og_image

    def restore_stretch(self, img, size):
        img = cv2.resize(img, (size[1], size[0]))
        return img

    def restore_size(self, img, size):
        if self.resize_mode == "pad": original_img = self.restore_pad(img, size)
        elif self.resize_mode == "stretch": original_img = self.restore_stretch(img, size)
        else: raise Exception(f"Unknown option: {self.resize_mode}")
        return original_img


    def __len__(self):
        return len(self.data_reader)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data_reader[idx]
        img, size = self.standardize_size(img)
        if self.post_resize_transform is not None:
            img = self.post_resize_transform(img)

        return {"size":size, "image":img}