

def tensor_to_cycle_gan_colour(in_image):
    return (in_image*2.0)-1

def cycle_gan_to_tensor_colour(in_image):
    return (in_image+1)/2