from collections import OrderedDict
import os
from typing import Any
import torch
import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models import create_model
from pytorch_CycleGAN_and_pix2pix.data import create_dataset
from pytorch_CycleGAN_and_pix2pix.options.test_options import TestOptions
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.util.visualizer import save_images
from pytorch_CycleGAN_and_pix2pix.util import html

from Helpers.image_resize_loader import ImageStandardizeDataLoader, ImageStandardizer
from Helpers.ab_loader import Custom_AB_Loader, Aligned_Class_Unaligned_Data_AB_Loader
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour, custom_cgan_train, inject_time_arg
import torch
import os
from tqdm import tqdm


def image_loader_tf(img, standardizer):
    img, _ = standardizer.standardize_size(img)
    img = openCV_to_tensor(img)
    img = tensor_to_cycle_gan_colour(img)
    return img

def original_img_restore(img, size, standardize:ImageStandardizer):
    size = size.tolist()
    img = cycle_gan_to_tensor_colour(img)
    img = tensor_to_openCV(img)
    img = standardizer.restore_size(img, size)

    return img

def inject_mode_arg(opt):
    mode = "stretch"
    if opt.dataroot == "pad": mode = "pad"
    return mode


def set_options():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    
    opt.batch_size = 1  
    opt.checkpoints_dir = "../Checkpoints/cycleGAN"
    opt.no_dropout = True
    return opt



def experiment_full_cycle(opt: Any, ab_loader: Aligned_Class_Unaligned_Data_AB_Loader, standardizer: ImageStandardizer):
    assert(opt.model=="cycle_gan")
    results_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))


    #dataset_batched = torch.utils.data.DataLoader(ab_loader, num_workers=opt.num_threads, batch_size=opt.batch_size) # type: ignore

    # Model + Data setup
    model = create_model(opt)
    model.setup(opt)
   
    # Web Output
    web_dir = results_dir
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    
    # Test Model
    model.eval()
    for class_idx in range(len(ab_loader.classes)):
        print(f"\t{ab_loader.classes[class_idx]}")
        indices = ab_loader.get_random_indices_of_class(class_idx, opt.num_test)
        for i in tqdm(indices):

            data = ab_loader[i]
            for k in ["A", "B"]:
                data[k] = data[k].unsqueeze(0)
            
            # Input
            model.set_input(data) 
            model.test()
            
            # Results
            visuals = model.get_current_visuals() 
            
            # Save Results f
            for j in range(opt.batch_size):    
                visuals_splice = OrderedDict()
                for k, img_root in [("real_A", "A"), ("fake_B", "A"), ("rec_A", "A"), ("real_B", "B"), ("fake_A", "B"), ("rec_B", "B")]:
                    img = visuals[k][j]
                    img = original_img_restore(img, data[f"{img_root}_shape"], standardizer)
                    visuals_splice[k] = tensor_to_cycle_gan_colour(openCV_to_tensor(img)).unsqueeze(0)
                
                save_images(webpage, visuals_splice, [f"{ab_loader.classes[class_idx]} {i}.png"], aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    
    webpage.save()


if __name__=="__main__":
    opt = set_options()
    standardizer= ImageStandardizer(256, 256)


    dl = Aligned_Class_Unaligned_Data_AB_Loader("../Dataset/Generated/Poses", "game-output-{id}.jpg", "movie-output-{id}.jpg", tf=lambda x: image_loader_tf(x, standardizer))

    opt.model="cycle_gan"
    opt.name = "better_patch_pad"
    opt.num_test = 10
    print("Pad")
    experiment_full_cycle(opt, dl, standardizer)
    opt.name = "better_patch_stretch"
    opt.num_test = 10
    print("Stretch")
    experiment_full_cycle(opt, dl, standardizer)
    
