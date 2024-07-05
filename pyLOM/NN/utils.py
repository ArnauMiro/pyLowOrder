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

class betaLinearScheduler:
	"""Beta schedule, linear growth to max value
	Args:
	   start_value (float): initial value of beta
	   end_value (float): final value of beta
	   warmup (int): number of epochs to reach final value"""

	def __init__(self, start_value, end_value, start_epoch, warmup):
		self.start_value = start_value
		self.end_value   = end_value
		self.start_epoch = start_epoch
		self.warmup      = warmup

	def getBeta(self, epoch):
		if epoch < self.start_epoch:
			return 0
		else:
			if epoch < self.warmup:
				beta = self.start_value + (self.end_value-self.start_value)*(epoch-self.start_epoch)/(self.warmup-self.start_epoch)
				return beta
			else:
				return self.end_value

class Dataset(torch_dataset):
	def __init__(self, vars, nx, ny, time, device='cpu', transform=True):
		self._nx = nx
		self._ny = ny
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
		mean = np.zeros((self._n_channels,self._nx*self._ny),dtype=float)
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
		mean = np.zeros((self._n_channels,self._nx*self._ny),dtype=float)
		maxi = np.zeros((self._n_channels,self.nt),dtype=float)
		for ichan in range(self._n_channels):
			var           = vars[ichan]
			maxi[ichan,:] = np.max(np.abs(var),axis=2)
			data.append(var/maxi[ichan,:])
		return data, mean, maxi
	
	def _get_data(self, vars):
		data = [] #tuple(None for _ in range(self._n_channels))
		mean = np.zeros((self._n_channels,self._nx*self._ny),dtype=float)
		maxi = np.zeros((self._n_channels,),dtype=float)
		for ichan in range(self._n_channels):
			var           = vars[ichan]
			mean[ichan,:] = temporal_mean(var)
			maxi[ichan]   = np.max(np.abs(var))
			data.append(var)
		return data, mean, maxi
	
	def crop(self, nx, ny, n0x, n0y):
		cropdata = []
		self._nx = nx
		self._ny = ny
		for ichannel in range(self._n_channels):
			isnap = self.data[ichannel]
			isnap = torch.Tensor(isnap)
			isnap = isnap.view(self.nt,n0x,n0y)
			isnap = TF.crop(isnap, top=0, left=0, height=nx, width=ny)
			cropdata.append(isnap.reshape(nx*ny,self.nt))
		self._data = cropdata

	def pad(self, nx, ny, n0x, n0y):
		paddata = []
		self._nx = n0x
		self._ny = n0y
		for ichannel in range(self._n_channels):
			isnap = self.data[ichannel]
			isnap = torch.Tensor(isnap)
			isnap = isnap.view(self.nt,nx,ny)
			isnap = F.pad(isnap, (0, n0y-ny, 0, n0x-nx), mode='constant', value=0)
			paddata.append(isnap.reshape(n0x*n0y,self.nt))
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
	
class Dataset3D(torch_dataset):
	def __init__(self, vars, nx, ny, nz, time, device='cpu', transform = True):
		self._nx = nx
		self._ny = ny
		self._nz = nz
		self._time = time
		self._device = device
		self._n_channels = len(vars)
		if transform:
			self._data, self._mean, self._max = self._normalize(vars)
		else:
			self._data, self._mean, self._max = self._get_data(vars)

	def __len__(self):
		return len(self._time)

	def __getitem__(self, index):
		snap = torch.Tensor([])
		for ichannel in range(self._n_channels):
			isnap = self.data[ichannel][:,index]
			isnap = torch.Tensor(isnap)
			isnap = isnap.view(1,self.nx,self.ny,self.nz)
			snap = torch.cat((snap,isnap), dim=0)
		return snap.to(self._device)
	
	@property
	def nx(self):
		return self._nx
	
	@property
	def ny(self):
		return self._ny
	
	@property
	def nz(self):
		return self._nz
	
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
	
	def _normalize(self, vars):
		data = []
		mean = np.zeros((self._n_channels,self._nx*self._ny*self._nz),dtype=float)
		maxi = np.zeros((self._n_channels,),dtype=float)
		for ichan in range(self._n_channels):
			var           = vars[ichan]
			mean[ichan,:] = temporal_mean(var)
			ifluc         = subtract_mean(var,mean[ichan,:])
			maxi[ichan]   = np.max(np.abs(ifluc))
			data.append(ifluc)
		return data, mean, maxi
	
	def _get_data(self, vars):
		data = [] 
		mean = np.zeros((self._n_channels,self._nx*self._ny*self._nz),dtype=float)
		maxi = np.zeros((self._n_channels,),dtype=float)
		for ichan in range(self._n_channels):
			var           = vars[ichan]
			#mean[ichan,:] = temporal_mean(var)
			#maxi[ichan]   = np.max(np.abs(var))
			data.append(var)
		return data, mean, maxi
	
	def crop(self, nx, ny, nz, n0x, n0y, n0z):

		## Crop for 3D data
		cropdata = []
		self._nx = nx
		self._ny = ny
		self._nz = nz
		for ichannel in range(self._n_channels):
			crops = []
			for t in range(self.nt):
				isnap = self.data[ichannel][:,t]
				isnap = torch.Tensor(isnap)
				isnap = isnap.view(1,n0x,n0y,n0z)
				isnap_cropped = torch.zeros(1, nx, ny, nz)
				xy_plane = torch.zeros(1,n0x,n0y)
				for z in range(n0z): 
					xy_plane = isnap[:,:,:,z]
					isnap_cropped[:,:,:,z] = TF.crop(xy_plane, top=0, left=0, height=nx, width=ny)
				crops.append(isnap_cropped.reshape(nx*ny*nz,1))
			crops = torch.cat(crops, dim=1)
			cropdata.append(crops)
		self._data = cropdata

	def pad(self, nx, ny, nz, n0x, n0y, n0z):

		## Pad for 3D data

		paddata = []
		self._nx = n0x
		self._ny = n0y
		self._nz = n0z
		for ichannel in range(self._n_channels):
			pads = []
			for t in range(self.nt):
				isnap = self.data[ichannel][:,t]
				isnap = torch.Tensor(isnap)
				isnap = isnap.view(1,nx,ny,nz)
				isnap_padded = torch.zeros(1, n0x, n0y, n0z)
				xy_plane = torch.zeros(1,nx,ny)
				for z in range(nz):
					xy_plane = isnap[:,:,:,z]
					isnap_padded[:,:,:,z] = F.pad(xy_plane, (0, n0y-ny, 0, n0x-nx), mode='constant', value = 0)
				pads.append(isnap_padded.reshape(n0x*n0y*n0z,1))
			pads = torch.cat(pads, dim = 1)
			paddata.append(pads)
		self._data = paddata
	
	def recover(self, data) :
		recovered_data = []
		for i in range(self._n_channels):
			recovered_data.append(data[i] + data[i].mean())
		return recovered_data
	
	def loader(self, batch_size=1,shuffle=True):
		#Compute number of snapshots
		loader = torch.utils.data.DataLoader(self, batch_size = batch_size, shuffle=shuffle)
		return loader

	def split_subdatasets(self, ptrain, pvali, batch_size=1, subdatasets=1):
			##Compute number of snapshots
			total_len = len(self)
			sub_len   = total_len // subdatasets
			len_train = int(ptrain*sub_len)
			len_vali  = int(pvali*sub_len)
			len_train = len_train + sub_len - (len_train + len_vali)
			
			##Select data
			train_subsets = []
			vali_subsets  = []
			for i in range(0, total_len, sub_len):
				sub_dataset = torch.utils.data.Subset(self, range(i, i + sub_len))
				train_subset, vali_subset = torch.utils.data.random_split(sub_dataset, (len_train, len_vali))
				train_subsets.append(train_subset)
				vali_subsets.append(vali_subset)
			combined_train_dataset = torch.utils.data.ConcatDataset(train_subsets)
			combined_vali_dataset = torch.utils.data.ConcatDataset(vali_subsets)
			train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True)
			vali_loader  = torch.utils.data.DataLoader(combined_vali_dataset, batch_size=batch_size, shuffle=True)

			return train_loader, vali_loader
