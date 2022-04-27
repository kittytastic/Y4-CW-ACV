import torch
import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions

from Helpers.image_resize_loader import ImageStandardizeDataLoader
from Helpers.ab_loader import Custom_AB_Loader
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour, custom_cgan_train, inject_time_arg

def map_data_to_required(data_batch):
    return {"A": data_batch["A_image"], "B": data_batch["B_image"], "A_paths": "none", "B_paths":"none"}

def image_loader_tf(img):
    img = openCV_to_tensor(img)
    img = tensor_to_cycle_gan_colour(img)
    return img

def inject_mode_arg(opt):
    mode = "stretch"
    if opt.dataroot == "pad": mode = "pad"
    return mode

if __name__=="__main__":

    opt = TrainOptions().parse()   # get training options
    
    mode = inject_mode_arg(opt)
    opt, target_hrs = inject_time_arg(opt)

    dl_A = ImageStandardizeDataLoader("../Dataset/Generated/HumanPatches/Games/Patches", 256, 256, resize_mode=mode, post_resize_transfrom=image_loader_tf)
    dl_B = ImageStandardizeDataLoader("../Dataset/Generated/HumanPatches/Movie/Patches", 256, 256, resize_mode=mode, post_resize_transfrom=image_loader_tf)
    dl = Custom_AB_Loader(dl_A, dl_B)

    
    ds = torch.utils.data.DataLoader(dl, num_workers=opt.num_threads, batch_size=opt.batch_size, shuffle=True) # type: ignore


    custom_cgan_train(opt, ds, target_hrs, map_data_to_required)
   
