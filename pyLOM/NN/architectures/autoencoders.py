#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# Autoencoder architecture for NN Module
#
# Last rev: 09/10/2024

import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

from   torch.utils.data        import DataLoader
from   torch.amp               import GradScaler, autocast
from   torch.utils.tensorboard import SummaryWriter
from   torchsummary            import summary

from   functools               import reduce
from   operator                import mul

from   ..                      import DEVICE
from   ...utils                import cr, pprint


## Wrapper of a variational autoencoder
class Autoencoder(nn.Module):
    r"""
    Autoencoder class for neural network module. The model is based on the PyTorch.

    Args:
        latent_dim (int): Dimension of the latent space.
        in_shape (tuple): Shape of the input data.
        input_channels (int): Number of input channels.
        encoder (torch.nn.Module): Encoder model.
        decoder (torch.nn.Module): Decoder model.
        device (str): Device to run the model. Default is 'cuda' if available, otherwise 'cpu'.
    
    """

    def __init__(
        self,
        latent_dim: int,
        in_shape: tuple,
        input_channels: int,
        encoder: nn.Module,
        decoder: nn.Module,
        device: torch.device = DEVICE,
    ):
        super(Autoencoder, self).__init__()
        self.lat_dim  = latent_dim
        self.in_shape = in_shape
        self.inp_chan = input_channels
        self.N        = reduce(mul, in_shape)
        self.encoder  = encoder
        self.decoder  = decoder
        self._device  = device
        encoder.to(self._device)
        decoder.to(self._device)
        self.to(self._device)
        summary(self, input_size=(self.inp_chan, *self.in_shape),device=device)
      
    def _lossfunc(self, x, recon_x, reduction):
        return  F.mse_loss(recon_x.view(-1, self.N), x.view(-1, self.N),reduction=reduction)
    
    def forward(self, x):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def fit(
        self,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset = None,
        epochs: int = 100,
        callback=None,
        lr: float = 1e-3,
        BASEDIR: str = "./",
        reduction: str = "mean",
        lr_decay: float = 0.999,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ):
        r"""
        Train the autoencoder model. The logs are stored in the directory specified by BASEDIR with tensorboard format.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            eval_dataset (torch.utils.data.Dataset): Evaluation dataset.
            epochs (int): Number of epochs to train the model. Default is ``100``.
            callback: Callback object. Default is ``None``.
            lr (float): Learning rate. Default is ``1e-3``.
            BASEDIR (str): Directory to save the model. Default is ``"./"``.
            reduction (str): Reduction method for the loss function. Default is ``"mean"``.
            lr_decay (float): Learning rate decay. Default is ``0.999``.
            batch_size (int): Batch size. Default is ``32``.
            shuffle (bool): Whether to shuffle the dataset or not. Default is ``True``.
            num_workers (int): Number of workers for the Dataloader. Default is ``0``.
            pin_memory (bool): Pin memory for Dataloader. Default is ``True``.
        """
        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        train_data = DataLoader(train_dataset, **dataloader_params)
        eval_data  = DataLoader(eval_dataset, **dataloader_params)
        # Initialization
        prev_train_loss = 1e99
        writer = SummaryWriter(BASEDIR)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
        # Training loop
        for epoch in range(epochs):
            self.train()
            num_batches = 0
            tr_loss = 0
            for batch0 in train_data:
                batch = batch0.to(self._device)
                recon, _ = self(batch)
                loss = self._lossfunc(batch, recon, reduction)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
                num_batches += 1
            tr_loss /= num_batches
            # Validation phase
            if eval_dataset is not None:
                with torch.no_grad():
                    val_batches = 0
                    va_loss = 0
                    for val_batch0 in eval_data:
                        val_batch = val_batch0.to(self._device)
                        val_recon, _ = self(val_batch)
                        vali_loss = self._lossfunc(val_batch, val_recon, reduction)
                        va_loss += vali_loss.item()
                        val_batches += 1
                    va_loss /= val_batches
            # Logging
            writer.add_scalar("Loss/train", tr_loss, epoch + 1)
            writer.add_scalar("Loss/vali", va_loss, epoch + 1)
            # Early stopping
            if callback and callback.early_stop(va_loss, prev_train_loss, tr_loss):
                print(f'Early Stopper Activated at epoch {epoch}', flush=True)
                break
            prev_train_loss = tr_loss
            print(f'Epoch [{epoch+1} / {epochs}] average training loss: {tr_loss:.5e} | average validation loss: {va_loss:.5e}', flush=True)            
            # Learning rate scheduling
            scheduler.step()

        # Cleanup
        writer.flush()
        writer.close()
        torch.save(self.state_dict(), f'{BASEDIR}/model_state.pth')

    def reconstruct(self, dataset: torch.utils.data.Dataset):
        r"""
        Reconstruct the dataset using the trained autoencoder model. It prints the energy, mean, and fluctuation of the reconstructed dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to reconstruct.

        Returns:
            np.ndarray: Reconstructed dataset.
        """
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self._device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self._device)
                x_recon,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.N)

                    x = energy_batch[i].view(self.inp_chan * self.N)
                    xr = rec[:, :, i].view(self.inp_chan * self.N)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
    
    def latent_space(self, dataset: torch.utils.data.Dataset):
        r"""
        Compute the latent space of the elements of a given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to compute the latent space.

        Returns:
            np.ndarray: Latent space of the dataset elements.
        """
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,z = self(batch)
        return z

    def decode(self, z):
        r"""
        Decode the latent space to the original space.

        Args:
            z (np.ndarray): Element of the latent space.

        Returns:
            np.ndarray: Decoded latent space.
        """
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 

## Wrapper of a variational autoencoder
class VariationalAutoencoder(Autoencoder):
    r"""
    Variational Autoencoder class for neural network module. The model is based on the PyTorch.

    Args:
        latent_dim (int): Dimension of the latent space.
        in_shape (tuple): Shape of the input data.
        input_channels (int): Number of input channels.
        encoder (torch.nn.Module): Encoder model.
        decoder (torch.nn.Module): Decoder model.
        device (str): Device to run the model. Default is 'cuda' if available, otherwise 'cpu'.

    """
    def __init__(self, latent_dim, in_shape, input_channels, encoder, decoder, device=DEVICE):
        super(VariationalAutoencoder, self).__init__(latent_dim, in_shape, input_channels, encoder, decoder, device)

    def _reparamatrizate(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        epsilon = torch.randn_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        return  sample
             
    def _kld(self, mu, logvar):
        mum     = torch.mean(mu, axis=0)
        logvarm = torch.mean(logvar, axis=0)
        return 0.5*torch.sum(1 + logvar - mum**2 - logvarm.exp())
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self._reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z
    
    @cr('VAE.fit')
    def fit(
        self,
        train_dataset,
        eval_dataset=None,
        betasch=None,
        epochs=1000,
        callback=None,
        lr=1e-4,
        BASEDIR="./",
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    ):
        r"""
        Train the variational autoencoder model. The logs are stored in the directory specified by BASEDIR with tensorboard format.

        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset.
            eval_dataset (torch.utils.data.Dataset): Evaluation dataset.
            epochs (int): Number of epochs to train the model. Default is ``100``.
            callback: Callback object to change the value of beta during training. Default is ``None``.
            lr (float): Learning rate. Default is ``1e-3``.
            BASEDIR (str): Directory to save the model. Default is ``"./"``.
            reduction (str): Reduction method for the loss function. Default is ``"mean"``.
            lr_decay (float): Learning rate decay. Default is ``0.999``.
            batch_size (int): Batch size. Default is ``32``.
            shuffle (bool): Whether to shuffle the dataset or not. Default is ``True``.
            num_workers (int): Number of workers for the Dataloader. Default is ``0``.
            pin_memory (bool): Pin memory for Dataloader. Default is ``True``.
        """

        dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        train_data = DataLoader(train_dataset, **dataloader_params)
        eval_data  = DataLoader(eval_dataset, **dataloader_params)
        prev_train_loss = 1e99
        writer    = SummaryWriter(BASEDIR)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0, amsgrad=False if self._device == "cpu" else True, fused=False if self._device == "cpu" else True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*1e-3)
        scaler    = GradScaler()
        for epoch in range(epochs):
            ## Training
            self.train()
            tr_loss = 0
            mse     = 0
            kld     = 0
            beta    = betasch.getBeta(epoch) if betasch is not None else 0
            for batch0 in train_data:
                batch = batch0.to(self._device)
                optimizer.zero_grad()
                with autocast(device_type=self._device):
                    recon, mu, logvar, _ = self(batch)
                    mse_i = self._lossfunc(batch, recon, reduction='sum')
                    kld_i = self._kld(mu,logvar)
                    loss  = mse_i - beta*kld_i
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item()
                mse     += mse_i.item()
                kld     += kld_i.item()
            num_batches = len(train_data)
            tr_loss /= num_batches
            mse /= num_batches
            kld /= num_batches

            ## Validation
            self.eval()
            va_loss     = 0
            with torch.no_grad():
                for val_batch0 in eval_data:
                    val_batch = val_batch0.to(self._device)
                    with autocast(device_type=self._device):
                        val_recon, val_mu, val_logvar, _ = self(val_batch)
                        mse_i     = self._lossfunc(val_batch, val_recon, reduction='sum')
                        kld_i     = self._kld(val_mu,val_logvar)
                        vali_loss = mse_i - beta*kld_i
                    va_loss  += vali_loss.item()

            num_batches = len(eval_data)
            va_loss    /=num_batches
            writer.add_scalar("Loss/train",tr_loss,epoch+1)
            writer.add_scalar("Loss/vali", va_loss,epoch+1)
            writer.add_scalar("Loss/mse",  mse,    epoch+1)
            writer.add_scalar("Loss/kld",  kld,    epoch+1)

            if callback is not None:
                if callback.early_stop(va_loss, prev_train_loss, tr_loss):
                    pprint(0, 'Early Stopper Activated at epoch %i' %epoch, flush=True)
                    break
            prev_train_loss = tr_loss   
            pprint(0, 'Epoch [%d / %d] average training loss: %.5e (MSE = %.5e KLD = %.5e) | average validation loss: %.5e' % (epoch+1, epochs, tr_loss, mse, kld, va_loss), flush=True)
            # Learning rate scheduling
            scheduler.step()

        writer.flush()
        writer.close()
        torch.save(self.state_dict(), '%s/model_state' % BASEDIR)

    @cr('VAE.reconstruct')
    def reconstruct(self, dataset):
        r"""
        Reconstruct the dataset using the trained variational autoencoder model. It prints the energy, mean, and fluctuation of the reconstructed dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to reconstruct.

        Returns:
            np.ndarray: Reconstructed dataset.
        """
        ## Compute reconstruction and its accuracy
        num_samples = len(dataset)
        ek = np.zeros(num_samples)
        mu = np.zeros(num_samples)
        si = np.zeros(num_samples)
        rec = torch.zeros((self.inp_chan, self.N, num_samples), device=self._device)

        loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)

        with torch.no_grad():
            ## Energy recovered in reconstruction
            for energy_batch in loader:
                energy_batch = energy_batch.to(self._device)
                x_recon,_,_,_ = self(energy_batch)

                for i in range(num_samples):
                    x_recchan = x_recon[i]
                    rec[:, :, i] = x_recchan.view(self.inp_chan, self.N)

                    x = energy_batch[i].view(self.inp_chan * self.N)
                    xr = rec[:, :, i].view(self.inp_chan * self.N)

                    ek[i] = torch.sum((x - xr) ** 2) / torch.sum(x ** 2)
                    mu[i] = 2 * torch.mean(x) * torch.mean(xr) / (torch.mean(x) ** 2 + torch.mean(xr) ** 2)
                    si[i] = 2 * torch.std(x) * torch.std(xr) / (torch.std(x) ** 2 + torch.std(xr) ** 2)

        energy = (1 - np.mean(ek)) * 100
        print('Recovered energy %.2f' % energy)
        print('Recovered mean %.2f' % (np.mean(mu) * 100))
        print('Recovered fluct %.2f' % (np.mean(si) * 100))

        return rec.cpu().numpy()
  
    def correlation(self, dataset):
        r"""
        Compute the correlation between the latent variables of the given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to compute the correlation.

        Returns:
            np.ndarray: Correlation between the latent variables.
        """
        ##  Compute correlation between latent variables
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,_,_, z = self(batch)
            np.save('z.npy',z.cpu())
            corr = np.corrcoef(z.cpu(),rowvar=False)
        detR = np.linalg.det(corr)*100
        print('Orthogonality between modes %.2f' % (detR))
        return corr, detR#.reshape((self.lat_dim*self.lat_dim,))
    
    def modes(self):
        r"""
        Compute the modes of the latent space.

        Returns:
            np.ndarray: Modes of the latent space.
        """
        zmode = np.diag(np.ones((self.lat_dim,),dtype=float))
        zmodt = torch.tensor(zmode, dtype=torch.float32)
        zmodt = zmodt.to(self._device)
        modes = self.decoder(zmodt)
        mymod = np.zeros((self.N,self.lat_dim),dtype=float)
        modes = modes.cpu()
        for imode in range(self.lat_dim):
            modesr = modes[imode,0,:,:].detach().numpy()
            mymod[:,imode] = modesr.reshape((self.N,), order='C')
        return mymod.reshape((self.N*self.lat_dim,),order='C')

    def latent_space(self, dataset):
        r"""
        Compute the latent space of the elements of a given dataset.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to compute the latent space.

        Returns:
            np.ndarray: Latent space of the dataset elements.
        """
        # Compute latent vectors
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        with torch.no_grad():
            instant  = iter(loader)
            batch    = next(instant)
            batch    = batch.to(self._device)
            _,_,_, z = self(batch)
        return z

    def fine_tune(self, train_dataset, shape_, eval_dataset=None, epochs=1000, callback=None, lr=1e-4, BASEDIR='./', **dataloader_params):
        train_data = DataLoader(torch.from_numpy(train_dataset).to(torch.float32), **dataloader_params)
        eval_data  = DataLoader(torch.from_numpy(eval_dataset).to(torch.float32), **dataloader_params)
        prev_train_loss = 1e99
        writer    = SummaryWriter(BASEDIR)
        decoder_model = self.decoder
        optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr, weight_decay=0, amsgrad=False if self._device == "cpu" else True, fused=False if self._device == "cpu" else True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*1e-3)
        scaler    = GradScaler()
        
        for epoch in range(epochs):
            ## Training
            decoder_model.train()
            tr_loss = 0
            for batch0 in train_data:
                batch = batch0.to(self._device)
                optimizer.zero_grad()
                with autocast(device_type=self._device):
                    in_data = batch[:, :self.lat_dim]
                    recon   = decoder_model(in_data)
                    loss    = self._lossfunc(torch.reshape(batch[:, self.lat_dim:], recon.shape), recon, reduction='sum')
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item()
            
            num_batches = len(train_data)
            tr_loss /= num_batches
            
            ## Validation
            decoder_model.eval()
            va_loss     = 0
            with torch.no_grad():
                for val_batch0 in eval_data:
                    val_batch = val_batch0.to(self._device)
                    with autocast(device_type=self._device):
                        val_in_data   = val_batch[:, :self.lat_dim]
                        val_recon     = decoder_model(val_in_data)
                        vali_loss     = self._lossfunc(torch.reshape(batch[:, self.lat_dim:], val_recon.shape), val_recon, reduction='sum')
                        
                    va_loss  += vali_loss.item()

            num_batches = len(eval_data)
            va_loss    /=num_batches
            writer.add_scalar("Ft/Loss/train",tr_loss,epoch+1)
            writer.add_scalar("Ft/Loss/vali", va_loss,epoch+1)
            
            if callback is not None:
                if callback.early_stop(va_loss, prev_train_loss, tr_loss):
                    pprint(0, 'Early Stopper Activated at epoch %i' %epoch, flush=True)
                    break
            prev_train_loss = tr_loss   
            pprint(0, 'Epoch [%d / %d] average training loss: %.5e | average validation loss: %.5e' % (epoch+1, epochs, tr_loss, va_loss), flush=True)
            # Learning rate scheduling
            scheduler.step()

        writer.flush()
        writer.close()
        torch.save(decoder_model.state_dict(), '%s/decoder_state' % BASEDIR)
        
        self.decoder.load_state_dict(torch.load('%s/decoder_state' % BASEDIR, weights_only=True))

        return 0

    def decode(self, z):
        r"""
        Decode a latent space element to the original space.

        Args:
            z (np.ndarray): Element of the latent space.

        Returns:
            np.ndarray: Decoded latent space.
        """
        zt  = torch.tensor(z, dtype=torch.float32)
        var = self.decoder(zt)
        var = var.cpu()
        varr = np.zeros((self.N,var.shape[0]),dtype=float)
        for it in range(var.shape[0]):
            varaux = var[it,0,:,:].detach().numpy()
            varr[:,it] = varaux.reshape((self.N,), order='C')
        return varr 