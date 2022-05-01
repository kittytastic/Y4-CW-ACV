import sys
sys.path.append("../pytorch_CycleGAN_and_pix2pix")
sys.path.append("../")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model
from Helpers.image_dir import ImageDirReader
from Helpers.cgan import cycle_gan_to_tensor_colour, tensor_to_cycle_gan_colour
from Helpers.image_resize_loader import ImageStandardizer
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV, openCV_to_tensor
from Models.augmentation import bg_eval
from Helpers.ab_loader import Aligned_Class_Unaligned_Data_AB_Loader
from Helpers.cgan import get_default_test_opt_full_cycle
import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
import random
import os


MODEL_NAME = "better_patch_pad_new"
EPOCH = "20"
MODE="pad"

def show_tile(img, path):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imwrite(path, img)


def tensor_resize_resize(tensor, size, algo_s:ImageStandardizer, display_s: ImageStandardizer):
    size = size.tolist()
    img_cv = tensor_to_openCV(tensor)
    img_cv = algo_s.restore_size(img_cv, size)
    img_cv, _ = display_s.standardize_size(img_cv)
    return openCV_to_tensor(img_cv) 

def image_loader_tf(img, standardizer):
    img, _ = standardizer.standardize_size(img)
    img = openCV_to_tensor(img)
    img = tensor_to_cycle_gan_colour(img)
    return img

def original_img_restore(img, size, standardizer:ImageStandardizer):
    size = size.tolist()
    img = cycle_gan_to_tensor_colour(img)
    img = tensor_to_openCV(img)
    img = standardizer.restore_size(img, size)

    return img

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


def view_patch():
    h,w, = 3,4
    display_s = ImageStandardizer(256, 256)
    algo_s = ImageStandardizer(256, 256, resize_mode=MODE)
    dl = Aligned_Class_Unaligned_Data_AB_Loader("../../Dataset/Generated/Poses", "game-output-{id}.jpg", "movie-output-{id}.jpg", tf=lambda x: image_loader_tf(x, algo_s))

    img_ids = random.sample(range(len(dl)), h*w)
    all_batches = [dl[id] for id in img_ids]

    output = run_cgan(all_batches)
    A_pairs = []
    for idx, b in enumerate(output):
        A_pairs.append(tensor_resize_resize(b["A_real"], all_batches[idx]["A_shape"], algo_s, display_s))
        A_pairs.append(tensor_resize_resize(b["A_fake"], all_batches[idx]["A_shape"], algo_s, display_s))
    
    B_pairs = []
    for idx, b in enumerate(output):
        B_pairs.append(tensor_resize_resize(b["B_real"], all_batches[idx]["B_shape"], algo_s, display_s))
        B_pairs.append(tensor_resize_resize(b["B_fake"], all_batches[idx]["B_shape"], algo_s, display_s))


    full_batch = torch.stack(A_pairs+B_pairs, dim=0)

    grid = torchvision.utils.make_grid(full_batch, nrow=2*w)
    show_tile(tensor_to_openCV(grid), "../../Artifacts/Q2_3.jpg")



if __name__=="__main__":
    view_patch()