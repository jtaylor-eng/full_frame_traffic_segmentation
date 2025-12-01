"""
PROJECT UTILITIES

Functionality to:
 - get sample images from a directory
 - distance function
 - ...

"""

#imports I needed in my homeworks
from skimage import filters
from skimage import io, color, transform
import numpy as np
import torch
from scipy.ndimage import convolve, maximum_filter
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') #avoid annoying imshow deprecated
import math
from pathlib import Path
import os
from typing import List, Tuple, Optional

"""
Image utils
"""
def get_sample_images(p: Path = Path('./sample_images')):
    try: return [io.imread(p / f) for f in os.listdir(p)]
    except: raise ValueError('cannot read sample images')

def plot_sample_images(images: List):
    try: 
        for img in images: io.imshow(img) ; plt.show()
    except: raise ValueError('cannot plot sample images')

def resize_images(images: List, target_size: Optional[Tuple]):
    return [transform.resize(img, target_size, anti_aliasing=True) for img in images]

"""
Distance functions. agnostic to torch and numpy
use compute_distance passing in the desired function in DISTANCE_FN 
 (supports L1, L2, cos)
"""
def euclidean_distance(p1, p2): 
    if isinstance(p1, torch.Tensor): return torch.norm(p1-p2, dim=-1) #torch

    return np.sqrt(np.sum((p1.astype(float)-p2.astype(float))**2, axis=-1))

def manhattan_distance(p1, p2):
    if isinstance(p1, torch.Tensor): return torch.sum(torch.abs(p1-p2), dim=-1)

    return np.sum(np.abs(p1.astype(float)-p2.astype(float)), axis=-1)

def cosine_distance(p1, p2):
    torch_obj = isinstance(p1, torch.Tensor)

    dot_prod = (p1*p2).sum(dim=-1) if torch_obj else np.sum(p1*p2, axis=-1)
    norm_p1 = torch.norm(p1, dim=-1, keepdim=True) if torch_obj else np.linalg.norm(p1, axis=-1, keepdims=False)
    norm_p2 = torch.norm(p2, dim=-1, keepdim=True) if torch_obj else np.linalg.norm(p2, axis=-1, keepdims=False) 

    return 1 - dot_prod / (norm_p1 * norm_p2 + 1e-8) 

DISTANCE_FNS = {'L1': manhattan_distance, 'L2': euclidean_distance, 'cos': cosine_distance}

def compute_distance(p1, p2, metric): 
    if isinstance(metric, str):
        if metric not in DISTANCE_FNS: raise ValueError('invalid distance metric')
        metric = DISTANCE_FNS[metric]

    return metric(p1,p2)

"""
other utils ...
"""
