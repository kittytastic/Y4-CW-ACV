from typing import Any

import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions

from Helpers.cgan import custom_cgan_train, inject_time_arg



if __name__ == '__main__':
    opt = TrainOptions().parse()

    opt, target_hrs = inject_time_arg(opt)
    dataset = create_dataset(opt)

    custom_cgan_train(opt, dataset, target_hrs, lambda x: x)    