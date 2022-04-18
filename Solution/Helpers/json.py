import json
import os.path
import concurrent.futures

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