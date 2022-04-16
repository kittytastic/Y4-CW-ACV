import torch

def init_torch()->torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count()>0:
        print(f"Using: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using: {device}")
    
    return device