import torch
import sys
sys.path.append("pytorch_CycleGAN_and_pix2pix")
from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions

from Helpers.image_resize_loader import ImageStandardizeDataLoader, ImageStandardizer
from Helpers.ab_loader import Custom_AB_Loader, Aligned_Class_Unaligned_Data_AB_Loader
from Helpers.images import tensor_to_openCV, openCV_to_tensor
from Helpers.cgan import tensor_to_cycle_gan_colour, cycle_gan_to_tensor_colour, custom_cgan_train, inject_time_arg


def image_loader_tf(img, snd: ImageStandardizer):
    img, _ = snd.standardize_size(img)
    img = openCV_to_tensor(img)
    img = tensor_to_cycle_gan_colour(img)
    return img

def inject_mode_arg(opt):
    mode = "stretch"
    if opt.dataroot == "pad": mode = "pad"
    return mode

if __name__=="__main__":

    #opt = TrainOptions().parse()   # get training options
    
    #mode = inject_mode_arg(opt)
    #opt, target_hrs = inject_time_arg(opt)

    image_standardizer = ImageStandardizer(256, 256)
    dl = Aligned_Class_Unaligned_Data_AB_Loader("../Dataset/Generated/Poses", "game-output-{id}.jpg", "movie-output-{id}.jpg", tf=lambda x: image_loader_tf(x, image_standardizer))


    dl[100]
    dl[1000]
    dl[20000]
    data = dl[32916]
    print(data)
    print(data["A"].shape)
    print(data["B"].shape)
