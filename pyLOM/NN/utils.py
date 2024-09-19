import numpy as np
import os
import math
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch.utils.data import Dataset as torch_dataset
from torch.utils.data import Subset
from torch import Generator, randperm, default_generator

from ..vmmath import temporal_mean, subtract_mean
from ..utils.cr import cr

from functools import reduce
from itertools import product, accumulate
from operator import mul
from typing import List, Optional, Tuple, cast, Sequence, Union
import warnings


class MinMaxScaler:
	"""
	Args:
			feature_range (Tuple): Desired range of transformed data. Default is ``(0, 1)``.
	"""
	def __init__(self, feature_range=(0, 1)):
		self.feature_range = feature_range

	def transform(
		self, variables: List[Union[np.ndarray, torch.tensor]]
	) -> List[Union[np.ndarray, torch.tensor]]:
		"""
		Scale variables to the range defined on `feature_range` using min-max scaling.
		Args:
				variables: List of variables to be scaled. The variables should be 2d numpy arrays or torch tensors.
		Returns:
				scaled_variables: List of scaled variables.
				variable_scaling_params: List of dictionaries containing the min and max values of each variable.
		"""

		scaled_variables = []
		variable_scaling_params = []
		for variable in variables:
			min_val = variable.min()
			max_val = variable.max()
			scaled_variable = (variable - min_val) / (max_val - min_val)
			# scale the variable to the desired feature_range
			scaled_variable = (
				scaled_variable * (self.feature_range[1] - self.feature_range[0])
				+ self.feature_range[0]
			)
			scaled_variables.append(scaled_variable)
			variable_scaling_params.append({"min": min_val, "max": max_val})

		self.variable_scaling_params = variable_scaling_params
		return scaled_variables

	def inverse_transform(self, variables: List[np.ndarray]) -> List[np.ndarray]:
		"""
		Inverse scale variables that have been scaled using min-max scaling.
		Args:
				variables: List of variables to be inverse scaled. The variables should be 2d numpy arrays.
				variable_scaling_params: List of dictionaries containing the min and max values of each variable.
		Returns:
				inverse_scaled_variables: List of inverse scaled variables.
		"""

		inverse_scaled_variables = []
		for variable, scaling_params in zip(variables, self.variable_scaling_params):
			min_val = scaling_params["min"]
			max_val = scaling_params["max"]
			inverse_scaled_variable = (variable - self.feature_range[0]) / (
				self.feature_range[1] - self.feature_range[0]
			)
			inverse_scaled_variable = inverse_scaled_variable * (max_val - min_val) + min_val
			inverse_scaled_variables.append(inverse_scaled_variable)
		return inverse_scaled_variables


class Dataset(torch.utils.data.Dataset):
	r"""
	Args:
			variables_out (Tuple): Tuple of variables to be used as output. Each variable should be a 2d numpy array or torch tensor. If only one varielbe wants to be provided, it should be passed as a tuple with one element. E.g. ``(variable,)``.
			mesh_shape (Tuple): Shape of the mesh. If not mesh is used and the data is considered as points, leave this as default. Default is ``(1,)``.
			variables_in (np.ndarray): Input variables. Default is ``None``.
			parameters (List[List[float]]): List of parameters to be used as input. Default is ``None``.
			inputs_scaler (MinMaxScaler): Scaler to scale the input variables. Default is ``None``.
			outputs_scaler (MinMaxScaler): Scaler to scale the output variables. Default is ``None``.
	"""

	def __init__(
		self,
		variables_out: Tuple,
		mesh_shape: Tuple = (1,),
		variables_in: np.ndarray = None,
		parameters: List[List[float]] = None,
		inputs_scaler=None,
		outputs_scaler=None,
	):
		self.parameters = parameters
		self.num_channels = len(variables_out)
		self.mesh_shape = mesh_shape
		if outputs_scaler is not None:
			variables_out = outputs_scaler.transform(variables_out)
		self.variables_out = self._process_variables_out(variables_out)
		if variables_in is not None:
			self.variables_in = self._process_variables_in(variables_in, parameters)
			if inputs_scaler is not None:
				self.variables_in = inputs_scaler.transform([self.variables_in])[0]

	def _process_variables_out(self, variables_out):
		if self.num_channels == 1:
			variables_out = torch.tensor(variables_out[0])
			variables_out = variables_out.unsqueeze(-1)
		else:
			variables_out = torch.cat(
				[torch.tensor(variable).unsqueeze(0) for variable in variables_out],
				dim=0,
			)  # (C, mul(mesh_shape), N)

		variables_out = variables_out.permute(2, 0, 1)  # (N, C, mul(mesh_shape))
		variables_out = variables_out.reshape(
			-1, self.num_channels, *self.mesh_shape
		)  # (N, C, *mesh_shape)
		if variables_out.shape[-1] == 1:  # (N, C, 1) -> (N, C)
			variables_out = variables_out.squeeze(-1)
		return variables_out.float()

	def _process_variables_in(self, variables_in, parameters):
		if parameters is None:
			variables_in = torch.tensor(variables_in, dtype=torch.float32)
			return variables_in
		variables_in = torch.tensor(variables_in, dtype=torch.float32)
		# parameters is a list of lists of floats. Each contains the values that will be repeated for each input coordinate
		# in some sense, it is like a cartesian product of the parameters with the input coordinates
		cartesian_product = list(product(*parameters))
		cartesian_product = torch.tensor(cartesian_product)
		variables_in_repeated = variables_in.repeat(len(cartesian_product), 1)
		cartesian_product = cartesian_product.repeat(len(variables_in), 1)
		return torch.cat([variables_in_repeated, cartesian_product], dim=1).float()

	def __len__(self):
		return len(self.variables_out)

	def __getitem__(self, idx):
		if self.variables_in is None:
			return self.variables_out[idx]
		return self.variables_in[idx], self.variables_out[idx]

	def get_splits(
		self,
		sizes,
		random=True,
		generator: Optional[Generator] = default_generator,
	):
		"""
		Randomly split a dataset into non-overlapping new datasets of given lengths.

		If a list of fractions that sum up to 1 is given,
		the lengths will be computed automatically as
		floor(frac * len(dataset)) for each fraction provided.

		After computing the lengths, if there are any remainders, 1 count will be
		distributed in round-robin fashion to the lengths
		until there are no remainders left.

		Optionally fix the generator for reproducible results, e.g.:

		Example:
				>>> # xdoctest: +SKIP
				>>> generator1 = torch.Generator().manual_seed(42)
				>>> generator2 = torch.Generator().manual_seed(42)
				>>> random_split(range(10), [3, 7], generator=generator1)
				>>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

		Args:
				dataset (Dataset): Dataset to be split
				sizes (sequence): lengths or fractions of splits to be produced
				generator (Generator): Generator used for the random permutation.
		"""
		if math.isclose(sum(sizes), 1) and sum(sizes) <= 1:
			subset_lengths: List[int] = []
			for i, frac in enumerate(sizes):
				if frac < 0 or frac > 1:
					raise ValueError(f"Fraction at index {i} is not between 0 and 1")
				n_items_in_split = int(
					math.floor(len(self) * frac)  # type: ignore[arg-type]
				)
				subset_lengths.append(n_items_in_split)
			remainder = len(self) - sum(subset_lengths)  # type: ignore[arg-type]
			# add 1 to all the lengths in round-robin fashion until the remainder is 0
			for i in range(remainder):
				idx_to_add_at = i % len(subset_lengths)
				subset_lengths[idx_to_add_at] += 1
			sizes = subset_lengths
			for i, length in enumerate(sizes):
				if length == 0:
					warnings.warn(
						f"Length of split at index {i} is 0. "
						f"This might result in an empty dataset."
					)

		# Cannot verify that dataset is Sized
		if sum(sizes) != len(self):  # type: ignore[arg-type]
			raise ValueError(
				"Sum of input lengths does not equal the length of the input dataset!"
			)
		if random:
			indices = randperm(sum(sizes), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
		else:
			indices = list(range(sum(sizes)))
		sizes = cast(Sequence[int], sizes)
		return [
			Subset(self, indices[offset - length : offset])
			for offset, length in zip(accumulate(sizes), sizes)
		]


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
		self.end_value = end_value
		self.start_epoch = start_epoch
		self.warmup = warmup

	def getBeta(self, epoch):
		if epoch < self.start_epoch:
			return 0
		else:
			if epoch < self.warmup:
				beta = self.start_value + (self.end_value - self.start_value) * (
					epoch - self.start_epoch
				) / (self.warmup - self.start_epoch)
				return beta
			else:
				return self.end_value


class DatasetOld(torch_dataset):
	def __init__(self, vars, inp_shape, time, device="cpu", transform=True):
		self._ndim = len(inp_shape)
		if self._ndim == 2:
			self._nh = inp_shape[0]
			self._nw = inp_shape[1]
		if self._ndim == 3:
			self._nd = inp_shape[0]
			self._nh = inp_shape[1]
			self._nw = inp_shape[2]
		self._N = reduce(mul, inp_shape)
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
			isnap = self.data[ichannel][:, index]
			isnap = torch.Tensor(isnap)
			isnap = (
				isnap.view(1, self._nh, self._nw)
				if self._ndim == 2
				else isnap.view(1, self._nd, self._nh, self._nw)
			)
			snap = torch.cat((snap, isnap), dim=0)
		return snap.to(self._device)

	@property
	def nd(self):
		return self._nd

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
		data = []  # tuple(None for _ in range(self._n_channels))
		mean = np.zeros((self._n_channels, self._N), dtype=float)
		maxi = np.zeros((self._n_channels,), dtype=float)
		for ichan in range(self._n_channels):
			var = vars[ichan]
			mean[ichan, :] = temporal_mean(var)
			ifluc = subtract_mean(var, mean[ichan, :])
			maxi[ichan] = np.max(np.abs(ifluc))
			data.append(ifluc)
		return data, mean, maxi

	def _normalize_min_max(self, vars):
		data = []  # tuple(None for _ in range(self._n_channels))
		mean = np.zeros((self._n_channels, self._N), dtype=float)
		maxi = np.zeros((self._n_channels, self.nt), dtype=float)
		for ichan in range(self._n_channels):
			var = vars[ichan]
			maxi[ichan, :] = np.max(np.abs(var), axis=2)
			data.append(var / maxi[ichan, :])
		return data, mean, maxi

	def _get_data(self, vars):
		data = []  # tuple(None for _ in range(self._n_channels))
		mean = np.zeros((self._n_channels, self._N), dtype=float)
		maxi = np.zeros((self._n_channels,), dtype=float)
		for ichan in range(self._n_channels):
			var = vars[ichan]
			mean[ichan, :] = temporal_mean(var)
			maxi[ichan] = np.max(np.abs(var))
			data.append(var)
		return data, mean, maxi

	def crop(self, shape, shape0):
		if len(shape) == 2:
			self._crop2D(self, shape[0], shape[1], shape0[0], shape0[1])
		if len(shape) == 3:
			self._crop3D(
				self, shape[0], shape[1], shape[2], shape0[0], shape0[1], shape0[2]
			)

	def pad(self, shape, shape0):
		if len(shape) == 2:
			self._pad2D(self, shape[0], shape[1], shape0[0], shape0[1])
		if len(shape) == 3:
			self._pad3D(
				self, shape[0], shape[1], shape[2], shape0[0], shape0[1], shape0[2]
			)

	def _crop2D(self, nh, nw, n0h, n0w):
		cropdata = []
		self._nh = nh
		self._nw = nw
		for ichannel in range(self._n_channels):
			isnap = self.data[ichannel]
			isnap = torch.Tensor(isnap)
			isnap = isnap.view(self.nt, n0h, n0w)
			isnap = TF.crop(isnap, top=0, left=0, height=nh, width=nw)
			cropdata.append(isnap.reshape(nh * nw, self.nt))
		self._data = cropdata

	def _pad2D(self, nh, nw, n0h, n0w):
		paddata = []
		self._nh = n0h
		self._nw = n0w
		for ichannel in range(self._n_channels):
			isnap = self.data[ichannel]
			isnap = torch.Tensor(isnap)
			isnap = isnap.view(self.nt, nh, nw)
			isnap = F.pad(isnap, (0, n0w - nw, 0, n0h - nh), mode="constant", value=0)
			paddata.append(isnap.reshape(n0h * n0w, self.nt))
		self._data = paddata

	def _crop3D(self, nd, nh, nw, n0d, n0h, n0w):
		## Crop for 3D data
		cropdata = []
		self._nd = nd
		self._nh = nh
		self._nw = nw
		for ichannel in range(self._n_channels):
			crops = []
			for t in range(self.nt):
				isnap = self.data[ichannel][:, t]
				isnap = torch.Tensor(isnap)
				isnap = isnap.view(1, n0d, n0h, n0w)
				isnap_cropped = torch.zeros(1, nd, nh, nw)
				xy_plane = torch.zeros(1, n0d, n0h)
				for z in range(n0w):
					xy_plane = isnap[:, :, :, z]
					isnap_cropped[:, :, :, z] = TF.crop(
						xy_plane, top=0, left=0, height=nd, width=nw
					)
				crops.append(isnap_cropped.reshape(nd * nh * nw, 1))
			crops = torch.cat(crops, dim=1)
			cropdata.append(crops)
		self._data = cropdata

	def _pad3D(self, nd, nh, nw, n0d, n0h, n0w):
		## Pad for 3D data
		paddata = []
		self._nd = n0d
		self._nh = n0h
		self._nw = n0w
		for ichannel in range(self._n_channels):
			pads = []
			for t in range(self.nt):
				isnap = self.data[ichannel][:, t]
				isnap = torch.Tensor(isnap)
				isnap = isnap.view(1, nd, nh, nw)
				isnap_padded = torch.zeros(1, n0d, n0h, n0w)
				xy_plane = torch.zeros(1, nd, nh)
				for z in range(nw):
					xy_plane = isnap[:, :, :, z]
					isnap_padded[:, :, :, z] = F.pad(
						xy_plane, (0, n0h - nh, 0, n0d - nd), mode="constant", value=0
					)
				pads.append(isnap_padded.reshape(n0d * n0h * n0w, 1))
			pads = torch.cat(pads, dim=1)
			paddata.append(pads)
		self._data = paddata

	def recover(self, data):
		recovered_data = []
		for i in range(self._n_channels):
			recovered_data.append(data[i] + data[i].mean())
		return recovered_data

	def loader(self, batch_size=1, shuffle=True):
		# Compute number of snapshots
		loader = torch.utils.data.DataLoader(
			self, batch_size=batch_size, shuffle=shuffle
		)
		return loader

	def split_subdatasets(self, ptrain, pvali, batch_size=1, subdatasets=1):
		##Compute number of snapshots
		total_len = len(self)
		sub_len = total_len // subdatasets
		len_train = int(ptrain * sub_len)
		len_vali = int(pvali * sub_len)
		len_train = len_train + sub_len - (len_train + len_vali)

		##Select data
		# Initialize lists to store subsets
		train_subsets = []
		vali_subsets = []
		# Split each third into training and validation
		for i in range(0, total_len, sub_len):
			sub_dataset = torch.utils.data.Subset(self, range(i, i + sub_len))
			train_subset, vali_subset = torch.utils.data.random_split(
				sub_dataset, (len_train, len_vali)
			)
			train_subsets.append(train_subset)
			vali_subsets.append(vali_subset)
		# Combine subsets from each third
		combined_train_dataset = torch.utils.data.ConcatDataset(train_subsets)
		combined_vali_dataset = torch.utils.data.ConcatDataset(vali_subsets)
		# Create DataLoaders
		train_loader = torch.utils.data.DataLoader(
			combined_train_dataset, batch_size=batch_size, shuffle=True
		)
		vali_loader = torch.utils.data.DataLoader(
			combined_vali_dataset, batch_size=batch_size, shuffle=True
		)

		return train_loader, vali_loader
