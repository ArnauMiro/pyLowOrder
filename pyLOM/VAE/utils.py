import torch
import numpy as np
import os

from torch.utils.data import Dataset as torch_dataset
from ..vmmath         import temporal_mean, subtract_mean
from ..utils.cr       import cr

def create_results_folder(RESUDIR):
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        print(f"Folder created: {RESUDIR}")
    else:
        print(f"Folder already exists: {RESUDIR}")

def select_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class Dataset(torch_dataset):
    @cr("Init VAE dataset")
    def __init__(self, var, nx, ny, time, transform = None):
        self._data, self._mean, self._max = self._normalize(var)
        self._nx                          = np.int(nx)
        self._ny                          = np.int(ny)
        self._time                        = time
        self.transform                    = transform
        
    def __len__(self):
        return len(self.time)
    
    def __getitem__(self, index):
        snap = torch.Tensor(self.data[:,index])
        snap = snap.resize_(1,self.nx,self.ny)        
        return snap
    
    @property
    def nx(self):
        return self._nx
    
    @property
    def ny(self):
        return self._ny
    
    @property
    def data(self):
        return self._data
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def max(self):
        return self._max
    
    @property
    def time(self):
        return self._time

    def _normalize(self, var):
        mean = temporal_mean(var)
        fluc = subtract_mean(var, mean)
        maxi = fluc.max()
        return np.array(fluc/fluc.max()), mean, maxi
    
    def recover(self, var):
        return var*self.max + np.tile(self.mean,(var.shape[1],1)).T
    
    def split(self, ptrain, pvali, batch_size=1):
        #Compute number of snapshots
        len_train = int(ptrain*len(self))
        len_vali  = int(pvali*len(self))
        len_train = len_train + len(self) - (len_train + len_vali)
        #Select data
        train, vali  = torch.utils.data.random_split(self,(len_train,len_vali))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        vali_loader  = torch.utils.data.DataLoader(vali, batch_size=batch_size, shuffle=True)
        return train_loader, vali_loader

class MultiChannelDataset(torch_dataset):
    def __init__(self, vars, nx, ny, time, transform=None):
        self._data = self._normalize(vars)
        self._nx = nx
        self._ny = ny
        self._time = time
        self.transform = transform
        self._n_channels = len(vars)

    def __len__(self):
        return len(self._time)

    def __getitem__(self, index):
        snap = torch.cat([torch.Tensor(self._data[i][:, index]).view(1, self._nx, self._ny) for i in range(self._n_channels)], dim=0)
        return snap

    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny

    @property
    def data(self):
        return self._data

    @property
    def time(self):
        return self._time

    @property
    def n_channels(self):
        return self._n_channels

    def _normalize(self, data_tuple):
        normalized_data = []
        for i in range(self._n_channels):
            data = data_tuple[i]
            mean = data.mean()
            fluc = data - mean
            maxi = fluc.max()
            normalized_data.append(fluc / maxi)
        return normalized_data

    def recover(self, data):
        recovered_data = []
        for i in range(self._n_channels):
            recovered_data.append(data[i] * data[i].max() + data[i].mean())
        return recovered_data

    def split(self, ptrain, pvali, batch_size=1):
        len_train = int(ptrain * len(self))
        len_vali = int(pvali * len(self))
        len_train = len_train + len(self) - (len_train + len_vali)
        train, vali = torch.utils.data.random_split(self, (len_train, len_vali))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        vali_loader = torch.utils.data.DataLoader(vali, batch_size=batch_size, shuffle=True)
        return train_loader, vali_loader