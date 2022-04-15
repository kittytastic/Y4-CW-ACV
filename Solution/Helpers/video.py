from typing import Optional, Tuple, Any
import numpy as np # type: ignore
import cv2
import os.path

class _VideoBase():
    @staticmethod
    def view_frame(frame: np.ndarray):
        cv2.imshow('frame',frame) # type: ignore
        cv2.waitKey(0)            # type: ignore

class VideoReader(_VideoBase):
    def __init__(self, video_path:str) -> None:
        super().__init__()
        self.path = video_path
        self.cap:Any = cv2.VideoCapture(video_path) # type: ignore
        self.num_frames:int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) #type: ignore
    
    def __del__(self):
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
        assert(frame_idx>0 and frame_idx<self.num_frames)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) # type: ignore
    
    def reset(self):
        self.seek(0)

    def __len__(self):
        return self.num_frames

    def __str__(self):
        w,h = self.get_resolution()
        fps = self.get_fps()
        return f"[Reader] {os.path.split(self.path)[-1]}: {w}x{h}@{fps:.2f}\tframe {self.get_pos()}/{len(self)}"

class VideoWriter(_VideoBase):
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