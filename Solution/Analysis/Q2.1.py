import sys
sys.path.append("../pytorch_CycleGAN_and_pix2pix")
sys.path.append("../")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model
from Helpers.image_dir import ImageDirReader
from Helpers.cgan import cycle_gan_to_tensor_colour
from Helpers.image_resize_loader import ImageStandardizer
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV, openCV_to_tensor
from Models.augmentation import bg_eval
from Helpers.cgan import get_default_test_opt_full_cycle
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import random
import os


MODEL_NAME = "games2movie_new"
EPOCH = "latest"

def show_tile(img, path):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imwrite(path, img)


def tensor_resize(tensor, image_s:ImageStandardizer):
    img_cv = tensor_to_openCV(tensor)
    img_cv, _ = image_s.standardize_size(img_cv)
    return openCV_to_tensor(img_cv) 

def run_cgan(all_batches):
    opt = get_default_test_opt_full_cycle(MODEL_NAME, epoch=EPOCH)
    opt.checkpoints_dir = os.path.join("../", opt.checkpoints_dir) 

    model = create_model(opt)
    model.setup(opt)
    results = []
    for data in all_batches:
         # Input
        model.set_input(data) 
        model.test()
        
        # Results
        visuals = model.get_current_visuals()
        result = {
                "A_real": visuals["real_A"].squeeze(0).cpu(),
                "A_fake": visuals["fake_B"].squeeze(0).cpu(),
                "B_real": visuals["real_B"].squeeze(0).cpu(),
                "B_fake": visuals["fake_A"].squeeze(0).cpu()
            }
        result = {k:cycle_gan_to_tensor_colour(img) for k, img in result.items()}
        results.append(result)

    return results


def view_background():
    h,w, = 4,3
    bg_A = ImageDirReader("../../Dataset/Generated/Background/testA", transform=bg_eval)
    img_ids = random.sample(range(len(bg_A)), h*w)
    A_imgs = [bg_A[id] for id in img_ids]
    bg_B = ImageDirReader("../../Dataset/Generated/Background/testB", transform=bg_eval)
    img_ids = random.sample(range(len(bg_B)), h*w)
    B_imgs = [bg_B[id] for id in img_ids]
   
    all_batches = [{"A": A.unsqueeze(0), "B": B.unsqueeze(0), "A_paths": [""], "B_paths": [""]} for A, B in zip(A_imgs, B_imgs)]

    image_s = ImageStandardizer(A_imgs[0].shape[1], A_imgs[0].shape[2])

    output = run_cgan(all_batches)
    A_pairs = []
    for b in output:
        A_pairs.append(b["A_real"])
        A_pairs.append(b["A_fake"])
    
    B_pairs = []
    for b in output:
        B_pairs.append(tensor_resize(b["B_real"], image_s))
        B_pairs.append(tensor_resize(b["B_fake"], image_s))


    full_batch = torch.stack(A_pairs+B_pairs, dim=0)

    grid = torchvision.utils.make_grid(full_batch, nrow=2*w)
    show_tile(tensor_to_openCV(grid), f"../../Artifacts/Q2_1_{EPOCH}.jpg")


if __name__=="__main__":
    view_background()