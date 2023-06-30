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