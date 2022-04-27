from torch import tensor
import random
import os
from typing import Any, Callable, Dict, List, Optional
import parse
import cv2

class Custom_AB_Loader:
    def __init__(self, A_loader, B_loader):
            self.A_size = len(A_loader)
            self.B_size = len(B_loader)

            self.A_loader = A_loader
            self.B_loader = B_loader

    def __getitem__(self, index):
        index_A = index%self.A_size
        index_B = random.randint(0, self.B_size - 1)
       
        A_data = self.A_loader[index_A]
        B_data = self.B_loader[index_B]

        merged_data = {**{f"A_{k}": v for k,v in A_data.items()}, **{f"B_{k}": v for k,v in B_data.items()}}

        return merged_data

    def __len__(self):
        return max(self.A_size, self.B_size)

def all_files_that_match_patten(dir:str, patten:str):
    comp_patten =parse.compile(patten)
    all_files =  [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    match_files = [f for f in all_files if comp_patten.parse(f) is not None]
    return match_files

def summary_big_dict(big_dict:Dict[Any, List[Any]]):
    summary_dict = {k:len(v) for k,v in big_dict.items()}
    return summary_dict

class Aligned_Class_Unaligned_Data_AB_Loader:
    def __init__(self, dir: str, A_patten:str, B_patten:str, tf: Optional[Callable[[Any], Any]]=None):
        self.dir = dir
        self.tf = tf
        self.A_patten = A_patten
        self.B_patten = B_patten
        self.classes = [d for d in os.listdir(self.dir) if os.path.isdir(os.path.join(self.dir, d))]

        self.files:Dict[str, Dict[str, List[str]]] = {}
        self.files["A"] = {c: all_files_that_match_patten(os.path.join(dir, c), self.A_patten) for c in self.classes}
        self.files["B"] = {c: all_files_that_match_patten(os.path.join(dir, c), self.B_patten) for c in self.classes}
        self.summary:Dict[str, Dict[str, int]] = {}
        self.summary["A"] = summary_big_dict(self.files['A'])
        self.summary["B"] = summary_big_dict(self.files['B'])
        self.total_len = {k: sum(self.summary[k].values()) for k in ["A", "B"]}
        self.primary_ds = "A" if self.total_len["A"]>self.total_len["B"] else "B"
        self.secondary_ds = "B" if self.total_len["A"]>self.total_len["B"] else "A"


    def get_index_class(self, idx):
        for c in self.classes:
            class_size = self.summary[self.primary_ds][c]
            if idx < class_size:
                return idx, c
            
            idx-=class_size

        raise IndexError

    def __getitem__(self, idx):
        idxs = {"A":0, "B":0}
        idxs[self.primary_ds], c = self.get_index_class(idx)
        idxs[self.secondary_ds] = random.randint(0, self.summary[self.secondary_ds][c]-1)

         
        img_paths = {f"{k}_paths": os.path.join(self.dir, c, self.files[k][c][idxs[k]]) for k in ["A", "B"]}
        imgs = {k:cv2.imread(img_paths[f"{k}_paths"]) for k in ["A", "B"]}
        imgs_sizes = {f"{k}_shape":tensor((imgs[k].shape[0], imgs[k].shape[1])) for k in ["A", "B"]}
        if self.tf is not None:
            tf_imgs = {k: self.tf(v) for k,v in imgs.items()}
        else:
            tf_imgs = imgs
        

        return {**tf_imgs, **img_paths, **imgs_sizes}

    def __len__(self):
        return self.total_len[self.primary_ds]