import os 
import json 
import torch
import random 
import numpy as np 
import pandas as pd 
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


def save_model(model, path):
    model.eval()
    try:
        torch.save(model.state_dict(), path)
        print('Successfully save weights to `{}`'.format(path))
    except Exception as e:
        print(e)
    return None 


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


class Meter:
    def __init__(self):
        self.reset()

    @property
    def names(self):
        return [k for k in self._history.keys()]

    def __getitem__(self, name):
        return self._history[name]

    def update(self, updates):
        for key, val in updates.items():
            self._history[key].append(val)

        return self

    def reset(self):
        self._history = defaultdict(list)

    def merge(self, meter):
        print(self._history)
        if len(self._history) == 0:
            self._history = meter._history
        else:
            for key, val in self._history.items():
                val += meter._history[key]
        return 

    def to_dataframe(self):
        return pd.DataFrame(self._history)
