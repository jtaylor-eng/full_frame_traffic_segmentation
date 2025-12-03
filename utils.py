"""
PROJECT UTILITIES

Functionality to:
 - get sample images from a directory
 - distance functions
 - KDE function
 - metrics (FPS, ARI, mIoU??)

"""

from skimage import io, transform
from sklearn.metrics import adjusted_rand_score
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import os
from typing import List, Tuple, Optional
import time
warnings.filterwarnings('ignore') #avoid annoying imshow deprecated

MINUTE: int = 60

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
KDE (guassian)
"""
def gaussian_KDE(distance, bandwith): 
    return np.exp(-0.5 * (distance / bandwith)**2)

KDE_FNS = {'gaussian': gaussian_KDE}

def compute_KDE(distance, bandwith, metric):
    if isinstance(metric, str):
        if metric not in KDE_FNS: raise ValueError('invalid kernel')
        metric = KDE_FNS[metric]

    return metric(distance,bandwith) 

"""
Metrics
"""
def compute_FPS(algorithm: callable, **alg_kwargs):
    start = time.perf_counter()
    end = start + MINUTE
    i = 0
    
    while time.perf_counter() < end:
        algorithm(**alg_kwargs)
        i+=1

    return  i / MINUTE

#NOTE: actually better to use Adjusted Rand Index for label agnosticism
def ARI(mask1, mask2):
    return adjusted_rand_score(mask1.flatten(), mask2.flatten())

#see ARI, better metric
def _IoU(mask1, mask2):
    pass

def mIoU(mask1, mask2):
    """Compute mIoU using masks

    NOTE: this will have to work for 'unlabelled' masks by computing 
    IoU with every 2 labels, and matching best IoU.

    This is because it is hard to specify that label 1 in image 1 will be 
    corresponding to label 1 in image 2 given the nature of our clustering 
    algorithms.  
    """
    pass
