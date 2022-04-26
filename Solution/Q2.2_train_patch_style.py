from Helpers.image_resize_loader import ImageStandardizeDataLoader
from Helpers.video import IOBase


if __name__=="__main__":
    dl = ImageStandardizeDataLoader("../Dataset/Generated/HumanPatches/Games/Patches", 800, 800, resize_mode="pad")

    data = dl[0]
    size, img = data["size"], data["image"]
    IOBase.view_frame(img)

    fixed_img = dl.restore_size(img, size)
    IOBase.view_frame(fixed_img)