import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from random import shuffle
from Helpers.image_resize_loader import ImageStandardizeDataLoader
from Helpers.ab_loader import Custom_AB_Loader
from Helpers.video import IOBase
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images, Visualizer
from pytorch_CycleGAN_and_pix2pix.util import html
from Helpers.video_loader import VideoLoader
from Helpers.video import IOBase, VideoWriter, VideoReader, DualVideoWriter
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour
from tqdm import tqdm
from collections import OrderedDict
import time
import torch

def map_data_to_required(data_batch):
    return {"A": data_batch["A_image"], "B": data_batch["B_image"], "A_paths": "none", "B_paths":"none"}

def train(opt, data_loader):
    opt.save_latest_freq = 1000
    opt.save_epoch_freq = 1
    opt.display_id = -1
    opt.model = "cycle_gan"
    opt.checkpoints_dir = "../Checkpoints/cycleGAN"

    dataset = data_loader
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            data=map_data_to_required(data)
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

def image_loader_tf(img):
    img = openCV_to_tensor(img)
    img = tensor_to_cycle_gan_colour(img)
    return img

if __name__=="__main__":

    opt = TrainOptions().parse()   # get training options
    
    mode = "stretch"
    if opt.dataroot == "pad": mode = "pad"

    dl_A = ImageStandardizeDataLoader("../Dataset/Generated/HumanPatches/Games/Patches", 256, 256, resize_mode=mode, post_resize_transfrom=image_loader_tf)
    dl_B = ImageStandardizeDataLoader("../Dataset/Generated/HumanPatches/Movie/Patches", 256, 256, resize_mode=mode, post_resize_transfrom=image_loader_tf)
    dl = Custom_AB_Loader(dl_A, dl_B)

    
    ds = torch.utils.data.DataLoader(dl, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True) # type: ignore

    train(opt, ds)
   
