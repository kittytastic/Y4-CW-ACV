from typing import Any
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
from Helpers.video import IOBase, VideoWriter, VideoReader, DualVideoWriter
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour
from tqdm import tqdm
from collections import OrderedDict


def set_options():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    opt.batch_size = 4  
    opt.checkpoints_dir = "./pytorch_CycleGAN_and_pix2pix/checkpoints"
    opt.model = "test"
    opt.no_dropout = True
    opt.name = "style_monet_pretrained" if opt.name == "monet" else "style_cezanne_pretrained"
    return opt


def experiment(opt: Any, video_loader: VideoLoader):
    results_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))
    video_name  = f"{opt.name}.mp4"

    # Model + Data setup
    model = create_model(opt)
    model.setup(opt)
    dataset = torch.utils.data.DataLoader(video_loader, num_workers=opt.num_threads, batch_size=opt.batch_size) # type: ignore
    input_meta = VideoReader(video_loader.video_path)

    # Web Output
    web_dir = results_dir
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # Video Output
    wr = DualVideoWriter(os.path.join(results_dir, video_name), like_video=input_meta)

    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # Test Model
    model.eval()
    i = 0
    for data in tqdm(dataset, total = opt.num_test//opt.batch_size):
        if i >= opt.num_test: break
        i+=opt.batch_size
        
        # Input
        tensor_image, numpy_image = data[0], data[1]
        image_names = [f"{video_loader.video_path.split('.')[-2]}:    frame {j}." for j in range(i-opt.batch_size, i)]
        cgan_data = {"A": tensor_image, "A_paths":image_names}
        model.set_input(cgan_data) 
        model.test()
        
        # Results
        visuals = model.get_current_visuals() 
        
        # Save Results
        fake_images = cycle_gan_to_tensor_colour(visuals["fake"]).detach().cpu()
        for j in range(opt.batch_size):
            fake_image = tensor_to_openCV(fake_images[j])
            fake_image = fake_image[0:wr.height]
            wr.write_dual_frame(numpy_image[j].numpy(), fake_image)
            
            visuals_splice = OrderedDict()
            visuals_splice["real"] = visuals["real"][j].unsqueeze(0)
            visuals_splice["fake"] = visuals["fake"][j].unsqueeze(0)
            save_images(webpage, visuals_splice, [image_names[j]], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    webpage.save()

if __name__=="__main__":
    print("---------- Q2.1 Frame Transfer - Eval ---------")
    opt = set_options()

    print("\n\n---------- Q2.1 Frame Transfer - Eval ---------")
    ds = VideoLoader("../Dataset/Train/Games/Video1.mp4", user_transform=tensor_to_cycle_gan_colour)
    opt.num_test = 10
    experiment(opt, ds)