import json
import os.path
import concurrent.futures
import os
import torch

class JsonDirWriter():
    def __init__(self, dir_path: str, prefix:str = "output", debug:bool=False):
        assert(os.path.isdir(dir_path))

        self.dir_path = dir_path
        self.prefix = prefix
        self.file_type = "json"
        self.counter = 0
        self.debug = debug

    def write_objects(self, objects):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for o in objects:
                executor.submit(self.write_atomic_object, object=o, counter=self.counter)
                self.counter +=1
    
    def write_atomic_object(self, object, counter):
        file_name =  f"{self.prefix}-{counter}.{self.file_type}"
        indent = 4 if self.debug else None
        json_s = json.dumps(object, indent=indent)
        with open(os.path.join(self.dir_path, file_name), "w+") as f:
            f.write(json_s)
        
    def write_object(self, object):
        self.write_atomic_object(object, self.counter) 
        self.counter+=1

class JsonDataLoader(torch.utils.data.Dataset): #type: ignore
    def __init__(self, json_dir:str):
        assert(os.path.isdir(json_dir))
        self.json_dir = json_dir
        self.file_type = "json"
        self.ordered = True
        self.refresh()

    def refresh(self):
        eligable_files =  [f for f in os.listdir(self.json_dir) if os.path.isfile(os.path.join(self.json_dir, f))]
        if self.file_type is not None:
            eligable_files =  [f for f in eligable_files if os.path.splitext(f)[-1].strip(".")==self.file_type]
        
        if self.ordered:
            ids =  [int(f.split("-")[1].split(".")[0]) for f in eligable_files]
            id_file = list(zip(ids, eligable_files))
            id_file.sort()
            eligable_files = [f for _, f in id_file]
        
        self.eligable_files = eligable_files 

    def __len__(self):
        return len(self.eligable_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with open(os.path.join(self.json_dir, self.eligable_files[idx])) as f:
            json_s = f.read()


        obj = json.loads(json_s)
        for k in obj.keys():
            if isinstance(obj[k], list):
                print("Chaning Key")
                obj[k] = torch.tensor(obj[k])

        return obj