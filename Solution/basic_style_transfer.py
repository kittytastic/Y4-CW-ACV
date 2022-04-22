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
from Helpers.video import IOBase, VideoWriter, VideoReader
from Helpers.images import tensor_to_openCV, openCV_to_tensor

def adjust_color(in_image):
    return (in_image*2.0)-1

def fix_colour_from_cgan(in_image):
    return (in_image+1)/2

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
    input_meta = VideoReader("../Dataset/Train/Games/Video1.mp4")

    wr = VideoWriter("../Dataset/TMP/tmp.mp4", like_video=input_meta)

    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options


    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        tensor_image, numpy_image = data[0], data[1]
        cgan_data = {"A": tensor_image, "A_paths":["nopath"]}
        model.set_input(cgan_data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
        fake_image = fix_colour_from_cgan(visuals["fake"]).detach().cpu().squeeze()
        fake_image = tensor_to_openCV(fake_image)
        fake_image = fake_image[0:wr.height]
        wr.write_frame(fake_image)
    
    webpage.save()  # save the HTML