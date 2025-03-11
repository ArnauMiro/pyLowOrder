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
from   ...utils.cr             import cr

class Decoder(nn.Module):
	def __init__(self, output_size, hidden_size, decoder_sizes, dropout):
		super(Decoder, self).__init__()
		decoder_sizes.insert(0, hidden_size)
		decoder_sizes.append(output_size)
		self.layers = nn.ModuleList()

		for i in range(len(decoder_sizes)-1):
			self.layers.append(nn.Linear(decoder_sizes[i], decoder_sizes[i+1]))
			if i != len(decoder_sizes)-2:
				self.layers.append(nn.Dropout(dropout))
				self.layers.append(nn.ReLU())

	def forward(self, output):
		for layer in self.layers:
			output = layer(output)
		return output

class SHRED(nn.Module):
	def __init__(self, input_size, output_size, device, total_sensors, hidden_size=64, hidden_layers=2, decoder_sizes=[350, 400], dropout=0.1, nconfigs=1, compile=False, seed=-1):
		'''
		SHRED model definition
		Inputs
			input size (e.g. number of sensors)
			output size (e.g. full-order variable dimension)
			size of LSTM hidden layers (default to 64)
			number of LSTM hidden layers (default to 2)
			list of decoder layers sizes (default to [350, 400])
			dropout parameter (default to 0)
		'''
		super(SHRED,self).__init__()
		np.random.seed(0) if seed == -1 else np.random.seed(seed)
		if compile:
			self.lstm    = torch.compile(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True), mode="max-autotune")
			self.decoder = torch.compile(Decoder(output_size, hidden_size, decoder_sizes, dropout), mode="max-autotune")
		else:
			self.lstm    = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True)
			self.decoder = Decoder(output_size, hidden_size, decoder_sizes, dropout)

		self.sensxconfig   = input_size
		self.nconfigs      = nconfigs
		self.hidden_layers = hidden_layers
		self.hidden_size = hidden_size
		self.configs = np.zeros((self.nconfigs, self.sensxconfig), dtype=int)
		for kk in range(self.nconfigs):
			self.configs[kk,:] = np.random.choice(total_sensors, size=self.sensxconfig, replace=False)
		
		self.device = device
		self.to(device)

	def forward(self, x):
		_, (output, _) = self.lstm(x)
		output = output[-1].view(-1, self.hidden_size)

		return self.decoder(output)

	def freeze(self):
		self.eval()
		for param in self.parameters():
			param.requires_grad = False

	def unfreeze(self):
		self.train()
		for param in self.parameters():
			param.requires_grad = True
	
	def _loss_func(self, x, recon_x, reduction):
		return F.mse_loss(x, recon_x, reduction=reduction)
	
	def _mre(self, x, recon_x):
		diff = (x-recon_x)*(x-recon_x)
		num  = torch.sqrt(torch.sum(diff, axis=1))
		den  = torch.sqrt(torch.sum(x*x, axis=1))
		return torch.mean(num/den)

	@cr.start('SHRED.fit')
	def fit(self, train_dataset, valid_dataset, batch_size=64, epochs=4000, optim=torch.optim.Adam, lr=1e-3, reduction='mean', verbose=False, patience=5):
		'''
		Neural networks training

		Inputs
			model (`torch.nn.Module`)
			training dataset (`torch.Tensor`)
			validation dataset (`torch.Tensor`)
			batch size (default to 64)
			number of epochs (default to 4000)
			optimizer (default to `torch.optim.Adam`)
			learning rate (default to 0.001)
			loss function (defalut to Mean Squared Error)
			loss value to print and return (default to Mean Relative Error)
			loss formatter for printing (default to percentage format)
			verbose parameter (default to False) 
			patience parameter (default to 5)
		'''

		train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
		optimizer = optim(self.parameters(), lr = lr)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*1e-4)
		valid_error_list = []
		patience_counter = 0
		best_params = self.state_dict()

		for epoch in range(1, epochs + 1):
			for k, data in enumerate(train_loader):
				self.train()
				outputs = self(data[0])
				optimizer.zero_grad()
				loss = self._loss_func(outputs, data[1], reduction)
				loss.backward()
				optimizer.step()
			scheduler.step()
			self.eval()
			with torch.no_grad():
				train_error = self._mre(train_dataset.Y, self(train_dataset.X))
				valid_error = self._mre(valid_dataset.Y, self(valid_dataset.X))
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
		train_error = self._mre(train_dataset.Y, self(train_dataset.X))
		valid_error = self._mre(valid_dataset.Y, self(valid_dataset.X))
		print("Training done: Training loss = %.2f Validation loss = %.2f \r" % (train_error*100, valid_error*100), flush=True)