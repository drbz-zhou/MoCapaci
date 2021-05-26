import numpy as np
import torch
from torch.utils.data import Dataset
from random import sample

import re

# from cvtorchvision import cvtransforms

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

class Dataload_fabric(Dataset):
    def __init__(self, input_path, idx_list, window_size, channel_size):
        self.data = np.empty(shape=[0,channel_size, window_size])
        self.label = np.empty(shape=[0,])
        self.label = self.label.astype(int)

        for num in idx_list:
            X_dir = input_path + 'P'+ str(num) + '_X.npy'
            Y_dir = input_path + 'P'+ str(num) + '_y.npy'

            print(X_dir, Y_dir)
            x = np.load(X_dir)
            y = np.load(Y_dir)
            x = x.transpose(0,2,1)
            y = y-1

            self.data = np.append(self.data, x, axis=0)
            self.label = np.append(self.label, y, axis=0)
        
        print(self.data.shape, self.label.shape)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        x = self.data[i]
        y = self.label[i]
        
        x = torch.from_numpy(x).float()
        # y = torch.from_numpy(y).float()
        return x, y
