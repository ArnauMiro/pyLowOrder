#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# SHRED architecture for NN Module
#
# Williams, J. P., Zahn, O., & Kutz, J. N. (2023). Sensing with shallow recurrent decoder networks. arXiv preprint arXiv:2301.12011.
#
# Last rev: 11/03/2025

import torch

import numpy               as np
import torch.nn            as nn
import torch.nn.functional as F

from   torch.utils.data    import DataLoader
from   ...utils.cr         import cr
from   .encoders_decoders  import ShallowDecoder
from   ..dataset           import Dataset
from ..utils.errors        import raiseError


class SHRED(nn.Module):
	r'''
	Shallow recurrent decoder (SHRED) architecture. For more information on the theoretical background of the architecture check the following reference
		Williams, J. P., Zahn, O., & Kutz, J. N. (2023). Sensing with shallow recurrent decoder networks. arXiv preprint arXiv:2301.12011.
	
	The model is based on the PyTorch library `torch.nn` (detailed documentation can be found at https://pytorch.org/docs/stable/nn.html). 

	In this implementation we assume that the output are always the POD coefficients of the full dataset.

	Args:
		output_size (int): Number of POD modes.
		device (torch.device): Device to use.
		total_sensors (int): Total number of sensors that will be used to ensamble the different configurations.
		hidden_size (int, optional): Dimension of the LSTM hidden layers (default: ``64``).
		hidden_layers (int, optional): Number of LSTM hidden layers (default: ``2``).
		decoder_sizes (list, optional): Integer list of the decoder layer sizes (default: ``[350, 400]``).
		input_size (int, optional): Number of sensor signals used as input (default: ``3``).
		dropouts (float, optional): Dropout probability for the decoder (default: ``0.1``).
		nconfigs (int, optional): Number of configurations to train SHRED on (default: ``1``).
		compile (bool, optional): Flag to compile the model (default: ``False``).
		seed (int, optional): Seed for reproducibility (default: ``-1``).
	'''
	def __init__(
			self, 
			output_size:int, 
			device:torch.device, 
			total_sensors:int, 
			hidden_size:int=64, 
			hidden_layers:int=2, 
			decoder_sizes:list=None, 
			input_size:int=3, 
			dropout:int=0.1, 
			nconfigs:int=1, 
			compile:bool=False, 
			seed:int=-1):
		super(SHRED,self).__init__()
		rng = np.random.default_rng(None if seed == -1 else seed)
		if decoder_sizes is None:
			decoder_sizes = [350, 400]
		if compile:
			self.lstm    = torch.compile(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True), mode="max-autotune")
			self.decoder = torch.compile(ShallowDecoder(output_size, hidden_size, decoder_sizes, dropout), mode="max-autotune")
		else:
			self.lstm    = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True)
			self.decoder = ShallowDecoder(output_size, hidden_size, decoder_sizes, dropout)

		self.sensxconfig   = input_size
		self.nconfigs      = nconfigs
		self.hidden_layers = hidden_layers
		self.hidden_size   = hidden_size
		self.configs = np.zeros((self.nconfigs, self.sensxconfig), dtype=int)
		for kk in range(self.nconfigs):
			self.configs[kk,:] = rng.choice(total_sensors, size=self.sensxconfig, replace=False)
		
		self.device = device
		self.to(device)

	def forward(self, x:torch.Tensor):
		r'''
		Do a forward evaluation of the data.

		Args:
			x (torch.Tensor): input data to the neural network.

		Returns:
			(torch.Tensor): Prediction of the neural network.
		'''
		_, (output, _) = self.lstm(x)
		output = output[-1].view(-1, self.hidden_size)

		return self.decoder(output)

	def freeze(self):
		r'''
		Freeze the model parameters to set it on inference mode.
		'''
		self.eval()
		for param in self.parameters():
			param.requires_grad = False

	def unfreeze(self):
		r'''
		Unfreeze the model parameters to set it on training mode.
		'''
		self.train()
		for param in self.parameters():
			param.requires_grad = True
	
	def _loss_func(self, x_true:torch.Tensor, x_pred:torch.Tensor, mod_scale:torch.Tensor, reduction:str):
		r'''
		Model loss function.

		Args:
			x_true (torch.Tensor): correct output.
			x_pred (torch.Tensor): neural network output.
			mod_scale (torch.Tensor): scaling of each POD coefficient according to its energy.
			reduction (str): type of reduction applied when doing the MSE.
		Returns:
			(double): Loss function
		'''
		return F.mse_loss(x_pred*mod_scale, x_true*mod_scale, reduction=reduction)

	def _mre(self, x_true:torch.Tensor, x_pred:torch.Tensor, mod_scale:torch.Tensor):
		r'''
		Mean relative error between the original and the SHRED reconstruction.

		Args:
			x_true (torch.Tensor): correct output.
			x_pred (torch.Tensor): neural network output.
			mod_scale (torch.Tensor): scaling of each POD coefficient according to its energy.
		Returns:
			(double): Mean relative error
		'''
		diff = (x_true-x_pred)*(x_true-x_pred)
		num  = torch.sqrt(torch.sum(diff, axis=0))
		den  = torch.sqrt(torch.sum(x_true*x_true, axis=0))
		return torch.sum(num/den*mod_scale/len(mod_scale))

	def _validate_dataset(self, dataset, name:str):
			r'''
			Check the shape contract before any permute or broadcast can hide a
			mismatch. variables_in must be (nsensors, nsnapshots, ndelays) and
			variables_out must be (nsnapshots, noutputs).

			Args:
				dataset (Dataset): dataset to validate.
				name (str): label used in the error messages.
			'''
			vin, vout = dataset.variables_in, dataset.variables_out
			if vin is None:
				raiseError('SHRED.fit: %s dataset has no variables_in' % name)
			if vin.ndim != 3:
				raiseError('SHRED.fit: %s variables_in must be 3D (nsensors, nsnapshots, ndelays), got %s'
						% (name, tuple(vin.shape)))
			if vout.ndim != 2:
				raiseError('SHRED.fit: %s variables_out must be 2D (nsnapshots, noutputs), got %s. '
						'A trailing singleton mesh axis is the usual cause: build the Dataset with '
						'mesh_shape=(1,) for point data, or squeeze it before calling fit.'
						% (name, tuple(vout.shape)))
			if vin.shape[1] != vout.shape[0]:
				raiseError('SHRED.fit: %s has %d input snapshots but %d output snapshots'
						% (name, vin.shape[1], vout.shape[0]))

	@cr('SHRED.fit')
	def fit(self, train_dataset: Dataset, valid_dataset: Dataset, batch_size:int=64, epochs:int=4000, optim:torch.optim.Optimizer=torch.optim.Adam, lr:float=1e-3, reduction:str='mean', verbose:bool=False, patience:int=5, mod_scale:torch.Tensor=None):
		r'''
		Fit of the SHRED model.

		Args:
			train_dataset (torch.utils.data.Dataset): training dataset.
			valid_dataset (torch.utils.data.Dataset): validation dataset.
			batch_size (int, optional): length of each training batch (default: ``64``).
			epochs (int, optional): number of epochs to extend the training (default: ``4000``).
			optim (torch.optim, optional): optimizer used (default: ``torch.optim.Adam``).
			lr (float, optional): learning rate (default: ``0.001``).
			verbose (bool, optional): define level of explicity on the output (default: ``False``). 
			patience (int, optional): epochs without improvements on the validation loss before stopping the training (default to 5).
		'''
		self._validate_dataset(train_dataset, 'training')
		self._validate_dataset(valid_dataset, 'validation')
		if train_dataset.variables_out.shape[1] != valid_dataset.variables_out.shape[1]:
			raiseError('SHRED.fit: training and validation datasets disagree on the number of outputs (%d against %d)' % 
			  (train_dataset.variables_out.shape[1],valid_dataset.variables_out.shape[1]))

		train_dataset.variables_in  = train_dataset.variables_in.permute(1,2,0).to(self.device)
		valid_dataset.variables_in  = valid_dataset.variables_in.permute(1,2,0).to(self.device)
		train_dataset.variables_out = train_dataset.variables_out.to(self.device)
		valid_dataset.variables_out = valid_dataset.variables_out.to(self.device)
		train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		optimizer    = optim(self.parameters(), lr = lr)
		scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*1e-4)
		valid_error_list = []
		patience_counter = 0
		best_params = self.state_dict()

		noutputs  = train_dataset.variables_out.shape[1]
		mod_scale = torch.ones((noutputs,), dtype=torch.float32, device=self.device) if mod_scale is None else mod_scale.to(self.device)
		if mod_scale.ndim != 1 or mod_scale.shape[0] != noutputs:
			raiseError('SHRED.fit: mod_scale must be 1D of length %d, got %s' % (noutputs, tuple(mod_scale.shape)))

		for epoch in range(1, epochs + 1):
			for k, data in enumerate(train_loader):
				self.train()
				outputs = self(data[0])
				optimizer.zero_grad()
				loss = self._loss_func(data[1], outputs, mod_scale, reduction)
				loss.backward()
				optimizer.step()
			scheduler.step()
			self.eval()
			with torch.no_grad():
				train_error = self._mre(train_dataset.variables_out, self(train_dataset.variables_in), mod_scale)
				valid_error = self._mre(valid_dataset.variables_out, self(valid_dataset.variables_in), mod_scale)
				valid_error_list.append(valid_error)
			if verbose == True:
				print("Epoch %i : Training loss = %.5e Validation loss = %.5e \r" % (epoch, train_error, valid_error), flush=True)
			if valid_error == torch.min(torch.tensor(valid_error_list)):
				patience_counter = 0
				best_params = self.state_dict().copy()
			else:
				patience_counter += 1
			if patience_counter == patience:
				break

		self.load_state_dict(best_params)
		train_error = self._mre(train_dataset.variables_out, self(train_dataset.variables_in), mod_scale)
		valid_error = self._mre(valid_dataset.variables_out, self(valid_dataset.variables_in), mod_scale)
		print("Training done: Training loss = %.2f Validation loss = %.2f \r" % (train_error*100, valid_error*100), flush=True)

	def save(self, path:str, scaler_path:str, podscale_path:str, sensors:np.array):
		r'''
		Save a SHRED configuration to a .pth file.

		Args:
			path (str): where the model will be saved.
			scaler_path (str): path to the scaler used to scale the sensor data.
			podscale_path (str): path to the scaler used for the POD coefficients.
			sensors (np.array): IDs of the sensors used for the current SHRED configuration.
		'''
		torch.save({
			'model_state_dict': self.state_dict(),
			'scaler_path'     : scaler_path,
			'podscale_path'   : podscale_path,
			'sensors'         : sensors,}, "%s.pth" % path)