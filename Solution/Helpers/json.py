import enum
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
        super().__init__()
        self.json_dir = json_dir
        self.file_type = "json"
        self.ordered = False 
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
                obj[k] = torch.tensor(obj[k])
        
        obj["file_name"] = self.eligable_files[idx]

        return obj


class JsonClassLoader(torch.utils.data.Dataset):
    def __init__(self, dir_path: str) -> None:
        assert(os.path.isdir(dir_path))
        super().__init__()

        self.dir_path = dir_path
        self.classes = [d for d in os.listdir(self.dir_path) if os.path.isdir(os.path.join(self.dir_path, d))]
        self.class_loaders = [JsonDataLoader(os.path.join(self.dir_path, d)) for d in self.classes]
        self.class_ids = {i:c for i,c in enumerate(self.classes)}
        self.class_limits = [0 for _ in range(len(self.classes))]
        self.class_starts = [0 for _ in range(len(self.classes))]

        end = 0
        for idx, cl in enumerate(self.class_loaders):
            self.class_starts[idx] = end
            end+=len(cl)
            self.class_limits[idx] = end

    def num_classes(self):
        return len(self.classes)

    def get_classes(self):
        return self.class_ids

    def __len__(self):
        return self.class_limits[-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_class_id = -1
        for i, cl in enumerate(self.class_limits):
            if idx<cl:
                target_class_id = i
                break
        
        if target_class_id == -1:
            raise IndexError

        relative_id = idx - self.class_starts[target_class_id]
        

        data = self.class_loaders[target_class_id][relative_id]
        data['class']=torch.tensor(target_class_id)

        return data

        