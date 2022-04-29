import torchvision
import numpy as np
import cv2
from PIL import Image

def tensor_to_openCV(tensor):
    frame = torchvision.transforms.ToPILImage()(tensor)
    frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def openCV_to_tensor(frame):
    col_correct = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torchvision.transforms.ToTensor()(col_correct)
    return tensor

def openCV_to_PIL(frame):
    col_correct = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(col_correct)

def PIL_to_tensor(frame):
    return torchvision.transforms.ToTensor()(frame)