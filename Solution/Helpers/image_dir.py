from typing import List, Optional, Tuple
from .video import IOBase, VideoReader
import os
import cv2
import numpy as np
import random

class ImageDirWriter(IOBase):
    def __init__(self, dir_path: str, prefix:str = "output", file_type:str="png"):
        assert(os.path.isdir(dir_path))

        self.dir_path = dir_path
        self.prefix = prefix
        self.file_type = file_type
        self.counter = 0
    
    def write_frame(self, frame):
        file_name =  f"{self.prefix}-{self.counter}.{self.file_type}"
        cv2.imwrite(os.path.join(self.dir_path, file_name), frame)
        self.counter+=1


class ImageDirReader(IOBase):
    def __init__(self, dir_path: str, file_type:Optional[str]=None, ordered:bool=True) -> None:
        assert(os.path.isdir(dir_path))

        self.dir_path = dir_path
        self.file_type = file_type
        self.ordered = ordered
        self.pos = 0
        self.refresh()


    def refresh(self):
        eligable_files =  [f for f in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, f))]
        if self.file_type is not None:
            eligable_files =  [f for f in eligable_files if os.path.splitext(f)[-1].strip(".")==self.file_type]
        
        if self.ordered:
            ids =  [int(f.split("-")[1].split(".")[0]) for f in eligable_files]
            id_file = list(zip(ids, eligable_files))
            id_file.sort()
            eligable_files = [f for _, f in id_file]
        
        self.eligable_files = eligable_files 

    def __iter__(self):
        self.counter=0
        return self

    def __next__(self)->np.ndarray:
        if self.pos<len(self.eligable_files):
            full_path = os.path.join(self.dir_path, self.eligable_files[self.pos])
            self.pos += 1
            return cv2.imread(full_path) 
        else:
            raise StopIteration

    def seek(self):
        self.pos = 0

    def random_sample(self, samples:int)->Tuple[List[np.ndarray], List[str]]:
        assert(samples<len(self.eligable_files))
        files_names = random.sample(self.eligable_files, samples)
        data = [cv2.imread(os.path.join(self.dir_path, f)) for f in files_names]
        return data, files_names

    def __len__(self):
        return len(self.eligable_files)



def _example():
    video_reader = VideoReader("../../Dataset/Test/Video1.mp4")
    writer = ImageDirWriter("./tmp")
    reader = ImageDirReader("./tmp", file_type="png")

    for _ in range(5):
        f = next(video_reader)
        writer.write_frame(f)

    reader.refresh()
    for f in reader:
        reader.view_frame(f)

    for frame, name in zip(*reader.random_sample(3)):
        reader.view_frame(frame)
        print(f"Random Sample: {name}")


if __name__=="__main__":
    _example()