"""
PROJECT UTILITIES

Functionality to:
 - get sample images from a directory
 - distance function
 - ...

"""


#imports I needed in my homeworks
from skimage import filters
from skimage import io, color
import numpy as np
from scipy.ndimage import convolve, maximum_filter
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') #avoid annoying imshow deprecated
import math
from pathlib import Path
import os
from typing import List

#Getting images
def GET_SAMPLE_IMAGES(p: Path = Path('./sample_images')):
    try: return [io.imread(p / f) for f in os.listdir(p)]
    except: raise ValueError('cannot read sample images')

def PLOT_SAMPLE_IMAGES(images: List):
    try: 
        for img in images: io.imshow(img) ; plt.show()
    except: raise ValueError('cannot plot sample images')

#Distance fns
def EUCLIDEAN_DIST(p1, p2): return np.linalg.norm(p1 - p2)
def MANHATTAN_DIST(p1, p2): return np.sum(np.abs(p1 - p2))
def COS_DIST(p1, p2):
    dot_prod = np.dot(p1, p2)
    norm_p1 = np.linalg(p1)
    norm_p2 = np.linalg(p2)

    if norm_p1 == 0 or norm_p2 == 0: return 1.0

    return 1 - (dot_prod / (norm_p1 * norm_p2))