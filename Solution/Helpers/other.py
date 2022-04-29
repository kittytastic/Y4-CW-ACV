from typing import List
import os 

def create_dirs(dirs: List[str]):
    for d in dirs:
        if not os.path.exists(d): os.makedirs(d)