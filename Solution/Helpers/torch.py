import torch

def init_torch()->torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # type: ignore
    if torch.cuda.device_count()>0: # type: ignore
        print(f"Using: {torch.cuda.get_device_name(0)}") # type: ignore
    else:
        print(f"Using: {device}")
    
    return device