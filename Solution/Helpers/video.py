from typing import List, Optional, Tuple, Any
import numpy as np # type: ignore
import cv2
import os.path
from torch import zeros
from torchvision.transforms import transforms

class IOBase():
    @staticmethod
    def view_frame(frame: np.ndarray):
        cv2.imshow('frame',frame) # type: ignore
        cv2.waitKey(0)            # type: ignore

class VideoBatchIter():
    def __init__(self, parent: 'VideoReader', batch_size: int) -> None:
        self.parent = parent
        self.batch_size = batch_size

    def __next__(self):
        frames:List[np.ndarray] = []
        for i in range(self.batch_size):
            frames.append(next(self.parent))
        
        return frames

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.parent)//self.batch_size

        

class VideoReader(IOBase):
    def __init__(self, video_path:str) -> None:
        self.finished_init = False
        super().__init__()
        self.path = video_path
        self.cap:Any = cv2.VideoCapture(video_path) # type: ignore
        self.num_frames:int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) #type: ignore
        self.finished_init = True
    
    def __del__(self):
        if self.finished_init:
            self.cap.release()

    def __next__(self)->np.ndarray:
        sucess, image = self.cap.read()
        if sucess:
            return image
        else:
            raise StopIteration

    def __iter__(self):
        return self
    
    def get_resolution(self)->Tuple[int, int]:
        w:int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # type:ignore
        h:int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # type: ignore
        return (w,h)
    
    def get_fps(self)->float:
        return self.cap.get(cv2.CAP_PROP_FPS) # type:ignore
    
    def get_pos(self)->int:
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) # type: ignore
    
    def seek(self, frame_idx:int):
        assert(frame_idx>=0 and frame_idx<self.num_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # type: ignore
    
    def reset(self):
        self.seek(0)

    def batch_iter(self, batch_size:int):
        return VideoBatchIter(self, batch_size)

    @staticmethod
    def frames_to_tensor(frames: List[np.ndarray]):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        all_frames = zeros((len(frames) ,frames[0].shape[2], frames[0].shape[0], frames[0].shape[1]))
        for idx, f in enumerate(frames):
            pf = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) # Convert from CV2's BGR to RGB
            frame_tensor = transform(pf) # Convert the image to tensor
            all_frames[idx] = frame_tensor


    def __len__(self):
        return self.num_frames

    def __str__(self):
        w,h = self.get_resolution()
        fps = self.get_fps()
        return f"[Reader] {os.path.split(self.path)[-1]}: {w}x{h}@{fps:.2f}\tframe {self.get_pos()}/{len(self)}"

class VideoWriter(IOBase):
    def __init__(self, output_name:str, resolution:Tuple[int, int]=(640, 480), fps:float = 29.97, like_video:Optional[VideoReader]=None) -> None:
        super().__init__()
        self.initialized = False
        if like_video is None:
            self.fps = fps
            self.width = resolution[0]
            self.height = resolution[1]
        else:
            self.fps = like_video.get_fps()
            self.width, self.height = like_video.get_resolution()
        
        _, file_type = os.path.splitext(output_name)
        file_fourccc_map = {".mp4": 'mp4v', ".avi":'XVID'}
        if file_type not in file_fourccc_map:
            raise Exception(f"Unsupported file type: {file_type}\nPlease choose from: {', '.join(file_fourccc_map.keys())}")
        
        self.output_name = output_name
        self.fourcc:Any = cv2.VideoWriter_fourcc(*file_fourccc_map[file_type]) # type: ignore
        self.out_stream:Any = cv2.VideoWriter(output_name, self.fourcc, self.fps, (self.width, self.height)) # type: ignore
        self.initialized=True
        self.pos =0
    
    def get_resolution(self)->Tuple[int, int]:
        return (self.width,self.height)
    
    def get_fps(self)->float:
        return self.fps
    
    def get_pos(self)->int:
        return self.pos
    
    def __del__(self):
        if self.initialized:
            self.out_stream.release()

    def write_frame(self, frame:np.ndarray):
        assert(frame.shape[0]==self.height)
        assert(frame.shape[1]==self.width)
        assert(frame.shape[2]==3)

        self.out_stream.write(frame)
        self.pos +=1
    
    def __str__(self):
        w,h = self.get_resolution()
        fps = self.get_fps()
        return f"[Writer] {os.path.split(self.output_name)[-1]}: {w}x{h}@{fps:.2f}\tframe {self.get_pos()}"


class DualVideoWriter():
    def __init__(self, output_name:str, like_video:VideoReader) -> None:
        self.video_writer = VideoWriter(output_name, like_video=like_video)
        self.width, self.height = self.video_writer.get_resolution()
    
    def write_dual_frame(self, f1, f2):
        assert(f1.shape[0]==self.height)
        assert(f1.shape[1]==self.width)
        assert(f1.shape[2]==3)
        assert(f2.shape[0]==self.height)
        assert(f2.shape[1]==self.width)
        assert(f2.shape[2]==3)
        half_width, half_height = self.width//2, self.height//2
        quater_height = half_height//2
        half_f1 = cv2.resize(f1, (half_width, half_height))
        half_f2 = cv2.resize(f2, (half_width, half_height))

        new_frame = np.zeros_like(f1)
        new_frame[quater_height:quater_height+half_height, 0:half_width,:]=half_f1
        new_frame[quater_height:quater_height+half_height, half_width:self.width,:]=half_f2
        self.video_writer.write_frame(new_frame)

def _example():
    save_frames = 100
    video = VideoReader('../../Dataset/Train/Games/Video1.mp4')
    video_writer = VideoWriter("output.avi", like_video=video)

    print(video)
    print(video_writer)

    for _ in range(save_frames):
        frame = next(video)
        video_writer.write_frame(frame)
    print()
    print(f"Copied: {save_frames} frames")
    print()
    print(video)
    print(video_writer)

if __name__=="__main__":
   _example() 