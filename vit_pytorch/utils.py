import os 
import torch
import random 
import numpy as np 
from collections import defaultdict


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except Exception as e:
            print(e)
    return None 


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())
    

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return None


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False 
    return None


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class Meter:
    def __init__(self):
        self.reset()

    @property
    def names(self):
        return [k for k in self._history.keys()]

    def update(self, updates):
        for key, val in updates.items():
            self._history[key].append(val)

        return self

    def reset(self):
        self._history = defaultdict(list)

    def __getitem__(self, name):
        return self._history[name]
