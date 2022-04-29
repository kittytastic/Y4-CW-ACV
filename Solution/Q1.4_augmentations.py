from Helpers.image_dir import ImageDirReader
from Helpers.images import openCV_to_PIL, PIL_to_tensor, tensor_to_openCV
import numpy as np
import torch
import torchvision
import cv2

def show_tile(img):
    max_h, max_w = 2000,2000
    h,w = img.shape[0], img.shape[1]
    sf_x, sf_y = max_h/h, max_w/w
    sf = min(sf_x, sf_y)
    img = cv2.resize(img, (int(w*sf), int(h*sf)))

    cv2.imshow('Tile', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bg_tf(img):
    img = openCV_to_PIL(img)
    transforms = torch.nn.Sequential(
        torchvision.transforms.ColorJitter(brightness=0.5),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomPerspective()
    )
    img = transforms(img)
    img = PIL_to_tensor(img)
    return img

if __name__=="__main__":
    h,w, = 5,5
    bg_A = ImageDirReader("../Dataset/Generated/Background/A_test", transform=tf)
    imgs = [bg_A[0] for _ in range(h*w)]
    batch = torch.stack(imgs, dim=0)

    grid = torchvision.utils.make_grid(batch, nrow=w)
    show_tile(tensor_to_openCV(grid))



    