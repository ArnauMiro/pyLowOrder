import torch
import numpy as np
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as F

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
    def __init__(self, vars, nh, nw, time, device='cpu', transform=True):
        self._nh = nh
        self._nw = nw
        self._time = time
        self._device = device
        self._n_channels = len(vars)
        if transform:
            self._data, self._mean, self._max = self._normalize_min_max(vars)
        else:
            self._data, self._mean, self._max = self._get_data(vars)

    def __len__(self):
        return len(self._time)

    def __getitem__(self, index):
        snap = torch.Tensor([])
        for ichannel in range(self._n_channels):
            isnap = self.data[ichannel][:,index]
            isnap = torch.Tensor(isnap)
            isnap = isnap.view(1,self._nh,self._nw)
            snap  = torch.cat((snap,isnap), dim=0)
        return snap.to(self._device)

    @property
    def nh(self):
        return self._nh

    @property
    def nw(self):
        return self._nw

    @property
    def data(self):
        return self._data

    @property
    def time(self):
        return self._time
    
    @property
    def nt(self):
        return self._time.shape[0]

    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def max(self):
        return self._max
       
    def _normalize(self, vars):
        data = [] #tuple(None for _ in range(self._n_channels))
        mean = np.zeros((self._n_channels,self._nh*self._nw),dtype=float)
        maxi = np.zeros((self._n_channels,),dtype=float)
        for ichan in range(self._n_channels):
            var           = vars[ichan]
            mean[ichan,:] = temporal_mean(var)
            ifluc         = subtract_mean(var,mean[ichan,:])
            maxi[ichan]   = np.max(np.abs(ifluc))
            data.append(ifluc)
        return data, mean, maxi
    
    def _normalize_min_max(self, vars):
        data = [] #tuple(None for _ in range(self._n_channels))
        mean = np.zeros((self._n_channels,self._nh*self._nw),dtype=float)
        maxi = np.zeros((self._n_channels,self.nt),dtype=float)
        for ichan in range(self._n_channels):
            var           = vars[ichan]
            maxi[ichan,:] = np.max(np.abs(var),axis=2)
            data.append(var/maxi[ichan,:])
        return data, mean, maxi
    
    def _get_data(self, vars):
        data = [] #tuple(None for _ in range(self._n_channels))
        mean = np.zeros((self._n_channels,self._nh*self._nw),dtype=float)
        maxi = np.zeros((self._n_channels,),dtype=float)
        for ichan in range(self._n_channels):
            var           = vars[ichan]
            mean[ichan,:] = temporal_mean(var)
            maxi[ichan]   = np.max(np.abs(var))
            data.append(var)
        return data, mean, maxi
    
    def crop(self, nh, nw, n0h, n0w):
        cropdata = []
        self._nh = nh
        self._nw = nw
        for ichannel in range(self._n_channels):
            isnap = self.data[ichannel]
            isnap = torch.Tensor(isnap)
            isnap = isnap.view(self.nt,n0h,n0w)
            isnap = TF.crop(isnap, top=0, left=0, height=nh, width=nw)
            cropdata.append(isnap.reshape(nh*nw,self.nt))
        self._data = cropdata

    def pad(self, nh, nw, n0h, n0w):
        paddata = []
        self._nh = n0h
        self._nw = n0w
        for ichannel in range(self._n_channels):
            isnap = self.data[ichannel]
            isnap = torch.Tensor(isnap)
            isnap = isnap.view(self.nt,nh,nw)
            isnap = F.pad(isnap, (0, n0w-nw, 0, n0h-nh), mode='constant', value=0)
            paddata.append(isnap.reshape(n0h*n0w,self.nt))
        self._data = paddata

    def recover(self, data):
        recovered_data = []
        for i in range(self._n_channels):
            recovered_data.append(data[i] + data[i].mean())
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
