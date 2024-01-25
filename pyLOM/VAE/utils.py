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

class DenoisingDataset(torch_dataset):
    def __init__(self, input, target, nx, ny, transform = None):
        self._input  = input
        self._target = target
        self._nx     = nx
        self._ny     = ny
        self.transform = transform

    def __len__(self):
        return len(self._input[0,:])
          
    def __getitem__(self, index):
        input  = torch.Tensor(self._input[:,index])
        input  = input.resize_(1,self.nx,self.ny) 
        target = torch.Tensor(self._target[:,index])
        target = target.resize_(1,self.nx,self.ny)         
        return input, target
    
    @property
    def nx(self):
        return self._nx

    @property
    def ny(self):
        return self._ny
       
    def loader(self, batch_size=1):
        loader  = torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=True)
        return loader

class MultiChannelDataset(torch_dataset):
    def __init__(self, vars, nx, ny, time, device='cpu', transform=None):
        self._nx = nx
        self._ny = ny
        self._time = time
        self.transform = transform
        self._device = device
        self._n_channels = len(vars)
        self._data, self._mean, self._max = self._normalize(vars)

    def __len__(self):
        return len(self._time)

    def __getitem__(self, index):
        snap = torch.Tensor([])
        for ichannel in range(self._n_channels):
            isnap = self.data[ichannel][:,index]
            isnap = torch.Tensor(isnap)
            isnap = isnap.view(1,self.nx,self.ny)
            snap  = torch.cat((snap,isnap), dim=0)
        return snap.to(self._device)

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

    def _normalize(self, vars):
        #data = tuple(None for _ in range(self._n_channels))
        data = []
        mean = np.zeros((self._n_channels,self._nx*self._ny),dtype=float)
        maxi = np.zeros((self._n_channels,),dtype=float)
        for ichan in range(self._n_channels):
            var           = vars[ichan]
            #mean[ichan,:] = temporal_mean(var)
            ifluc         = -subtract_mean(var,mean[ichan,:])
            maxi[ichan]   = np.max(np.abs(var))#-ifluc.max()
            data.append(var)#ifluc/maxi[ichan])
        return data, mean, maxi

    def recover(self, data):
        recovered_data = []
        for i in range(self._n_channels):
            recovered_data.append(data[i])# + data[i].mean())
        return recovered_data

    def loader(self, batch_size=1,shuffle=True):
        #Compute number of snapshots
        loader = torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
        return loader
    
    def split_subdatasets(self, ptrain, pvali, batch_size=1, subdatasets=1):
        ##Compute number of snapshots
        total_len = len(self)
        sub_len   = total_len // subdatasets
        len_train = int(ptrain*sub_len)
        len_vali  = int(pvali*sub_len)
        len_train = len_train + sub_len - (len_train + len_vali)
        
        ##Select data
        # Initialize lists to store subsets
        train_subsets = []
        vali_subsets  = []
        # Split each third into training and validation
        for i in range(0, total_len, sub_len):
            sub_dataset = torch.utils.data.Subset(self, range(i, i + sub_len))
            train_subset, vali_subset = torch.utils.data.random_split(sub_dataset, (len_train, len_vali))
            train_subsets.append(train_subset)
            vali_subsets.append(vali_subset)
        # Combine subsets from each third
        combined_train_dataset = torch.utils.data.ConcatDataset(train_subsets)
        combined_vali_dataset = torch.utils.data.ConcatDataset(vali_subsets)
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
        vali_loader  = torch.utils.data.DataLoader(combined_vali_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, vali_loader
