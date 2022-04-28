from typing import Callable, Any
import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images, Visualizer
import time
import argparse
import math

def tensor_to_cycle_gan_colour(in_image):
    return (in_image*2.0)-1

def cycle_gan_to_tensor_colour(in_image):
    return (in_image+1)/2


def inject_time_arg(opt):
    target_hrs = opt.init_gain
    opt.init_gain = 0.02
    assert(target_hrs!=opt.init_gain)
    print(f"Targeting: {target_hrs} hrs of training")
    return opt, target_hrs

def get_default_test_opt():
    return  argparse.Namespace(aspect_ratio=1.0, batch_size=1, checkpoints_dir='./checkpoints', crop_size=256, dataroot='non', dataset_mode='single', direction='AtoB', display_winsize=256, epoch='latest', eval=False, gpu_ids=[0], init_gain=0.02, init_type='normal', input_nc=3, isTrain=False, load_iter=0, load_size=256, max_dataset_size=math.inf, model='test', model_suffix='', n_layers_D=3, name='experiment_name', ndf=64, netD='basic', netG='resnet_9blocks', ngf=64, no_dropout=False, no_flip=False, norm='instance', num_test=50, num_threads=4, output_nc=3, phase='test', preprocess='resize_and_crop', results_dir='./results/', serial_batches=False, suffix='', use_wandb=False, verbose=False)

def apply_normal_test_opt(opt):
    opt.num_threads = 0 
    opt.batch_size = 1  
    opt.no_flip = True
    opt.eval = True
    opt.no_dropout = True
    opt.checkpoints_dir = "../Checkpoints/cycleGAN"
    opt.display_id = -1

    return opt

# Adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py
def custom_cgan_train(opt, data_loader, target_time:float, map_data_to_required:Callable[[Any], Any]):
    start_time = time.time()
    end_time = start_time + target_time*3600 - 5*60
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
        if time.time()>end_time:
            break
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if time.time()>end_time:
                break
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

    model.save_networks('latest')
    print("Finishing run, model saved")