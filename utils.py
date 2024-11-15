import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def untransform_video(videos):

    # Reverse normalization (assuming mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    mean = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1).to(videos.device)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1).to(videos.device)
    videos = videos * std + mean
    
    # Clip values to [0, 1] range
    videos = torch.clamp(videos, 0, 1)
    
    # Convert to uint8 for visualization
    videos = (videos * 255).to(torch.uint8)
    return videos