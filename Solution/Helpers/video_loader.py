from typing import Optional, Callable, Any
import torch.utils.data
from .video import VideoReader
import numpy as np
import math
import cv2
from torchvision.transforms import transforms

class VideoSubsectionReader(VideoReader):
    def __init__(self, video_path, start:int, end:int, user_transform:Optional[Callable[[Any], Any]]=None):
        super().__init__(video_path)
        self.start = start
        self.end = end
        self.pos = start
        self.tf = transforms.Compose([
            transforms.ToTensor()
        ])
        self.user_transforms = user_transform
    
    def __iter__(self):
        self.seek(self.start)
        return self

    def __next__(self) -> np.ndarray:
        '''
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
        else:
            worker_id = worker_info.id
        print(f"[{worker_id}]  Getting frame: {self.pos}")
        '''
        if self.pos>self.end:
            raise StopIteration
        self.pos += 1
        next_frame = super().__next__()
        col_correct = cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB)
        tensor_correct = self.tf(col_correct)
        if self.user_transforms is not None:
            tensor_correct = self.user_transforms(tensor_correct)
        return tensor_correct, next_frame 

class VideoLoader(torch.utils.data.IterableDataset):
    def __init__(self, video_path:str, user_transform:Optional[Callable[[Any], Any]] = None):
        super().__init__()
        self.start = 0
        self.video_info = VideoReader(video_path)
        self.end = len(self.video_info)
        self.video_info = None # cant multi thread with complex obj in loader
        self.video_path = video_path
        self.user_transform = user_transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        
        
        reader = VideoSubsectionReader(self.video_path, iter_start, iter_end, user_transform=self.user_transform)
        return iter(reader)