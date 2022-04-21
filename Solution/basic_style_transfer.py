import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")

import torch
import os
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images
from pytorch_CycleGAN_and_pix2pix.util import html
from Helpers.video_loader import VideoLoader
from Helpers.video import IOBase

def adjust_color(in_image):
    return (in_image*2.0)-1

if __name__=="__main__":
    print("---------- Basic Style Transfer ---------")

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.checkpoints_dir = "./pytorch_CycleGAN_and_pix2pix/checkpoints"
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))


    ds = VideoLoader("../Dataset/Train/Games/Video1.mp4", user_transform=adjust_color)
    dataset = torch.utils.data.DataLoader(ds, num_workers=opt.num_threads, batch_size=opt.batch_size) # type: ignore
    samples = 0


    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options


    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        #print(data[0].shape)
        #print(data[1].shape)
        cgan_data = {"A": data[0], "A_paths":["nopath"]}
        #print(data[0].numpy().shape)
        #print(data[1].numpy().shape)
        #IOBase.view_frame(data[0][0].numpy())
        #IOBase.view_frame(data[1][0].numpy())
        #print(data)
        #print(data["A"].shape)
        model.set_input(cgan_data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        #print(visuals)
        img_path = model.get_image_paths()     # get image paths
        #print(img_path)
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        #exit()
    webpage.save()  # save the HTML