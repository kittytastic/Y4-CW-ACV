from typing import Any

import sys
from Helpers.cgan import tensor_to_cycle_gan_colour
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions

from Helpers.cgan import custom_cgan_train, inject_time_arg
from Helpers.image_dir import ImageDirReader
from Helpers.ab_loader import Custom_AB_Loader
from Models.augmentation import game_bg_tf, movie_bg_tf
import torch
import random

class BG_model_AB_Loader:
    def __init__(self, A_loader, B_loader):
            self.A_size = len(A_loader)
            self.B_size = len(B_loader)

            self.A_loader = A_loader
            self.B_loader = B_loader

    def __getitem__(self, index):
        index_A = index%self.A_size
        index_B = random.randint(0, self.B_size - 1)
       
        A_data = self.A_loader[index_A]
        B_data = self.B_loader[index_B]

        return {"A":A_data, "B":B_data, "A_paths":"", "B_paths":""}

    def __len__(self):
        return max(self.A_size, self.B_size)

def game_tf(img):
    img = game_bg_tf(img)
    return tensor_to_cycle_gan_colour(img)

def movie_tf(img):
    img = movie_bg_tf(img)
    return tensor_to_cycle_gan_colour(img)

if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt, target_hrs = inject_time_arg(opt)

    dl_A = ImageDirReader("../Dataset/Generated/Background/trainA", transform=game_tf)
    dl_B = ImageDirReader("../Dataset/Generated/Background/trainB", transform=movie_tf)
    dl = BG_model_AB_Loader(dl_A, dl_B)
    
    ds = torch.utils.data.DataLoader(dl, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True)
    
    custom_cgan_train(opt, ds, target_hrs, lambda x: x)