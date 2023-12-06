import numpy as np
import random
import torch
import os
import sys
import os.path as osp

def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()