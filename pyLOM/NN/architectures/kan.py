import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ... import cr, pprint  # pyLOM/__init__.py
from ..  import DEVICE  # pyLOM/NN/__init__.py


class KAN(nn.Module):
    r"""
    KAN (Kolmogorov-Arnold Network) model for regression tasks. This model is based on https://arxiv.org/abs/2404.19756, inspired by the Kolmogorov-Arnold representation theorem.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        n_layers (int): The number of hidden layers.
        hidden_size (int): The number of neurons in the hidden layers.
        layer_type (nn.Module): The type of layer to use in the model. It can be one of the following: ``JacobiLayer``, ``ChebyshevLayer``.
        model_name (str): The name of the model.
        p_dropouts (float, optional): The dropout probability (default: ``0.0``).
        device (torch.device, optional): The device where the model is loaded (default: gpu if available).
        **layer_kwargs: Additional keyword arguments to pass to the layer type. For example, the order of the Taylor series or the degree of the Chebyshev polynomial.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int,
        hidden_size: int,
        layer_type,
        model_name: str,
        p_dropouts: float = 0.0,
        device: torch.device = DEVICE,
        **layer_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.layer_type = layer_type
        self.model_name = model_name
        self.p_dropouts = p_dropouts
        self.device = device

        # Hidden layers with dropout
        hidden_layers = []
        for _ in range(n_layers):
            hidden_layers.append(layer_type(hidden_size, hidden_size, **layer_kwargs))
            hidden_layers.append(nn.Dropout(p=p_dropouts))

        self.kan_layers = nn.ModuleList(hidden_layers)

        # Input and output layers
        self.input = layer_type(input_size, hidden_size, **layer_kwargs)
        self.output = layer_type(hidden_size, output_size, **layer_kwargs)

        self.to(self.device)

    def forward(self, x):
        x = self.input(x)
        for layer in self.kan_layers:
            x = layer(x)

        x = self.output(x)

        return x

    @cr("KAN.fit")
    def fit(
        self,
        train_dataset,
        eval_dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        lr_gamma: float,
        lr_scheduler_step: int,
        print_eval_rate: int = 2,
        loss_fn=nn.MSELoss(),
        save_logs_path=None,
    ):
        r"""
        Train the model using the provided training dataset. The model is trained using the Adam optimizer with the provided learning rate and learning rate decay factor.

        Args:
            train_dataset: The training dataset.
            eval_dataset: The evaluation dataset.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size.
            lr (float): The learning rate for the Adam optimizer.
            lr_gamma (float): The learning rate decay factor.
            lr_scheduler_step (int): The number of epochs to reduce the learning rate.
            print_eval_rate (int, optional): The model will be evaluated every ``print_eval_rate`` epochs and the losses will be printed (default: ``2``).
            loss_fn (torch.nn.Module, optional): The loss function (default: ``nn.MSELoss()``).
            save_logs_path (str, optional): Path to save the training and evaluation losses (default: ``None``).
        """

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        train_losses = torch.tensor([], device=self.device)
        test_losses = torch.tensor([], device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_scheduler_step, gamma=lr_gamma
        )
        if hasattr(self, "optimizer_state_dict"):
            self.optimizer.load_state_dict(self.optimizer_state_dict)
            del self.optimizer_state_dict

        if hasattr(self, "scheduler_state_dict"):
            self.scheduler.load_state_dict(self.scheduler_state_dict)
            del self.scheduler_state_dict
        

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = (
                    inputs.float().to(self.device),
                    targets.float().to(self.device),
                )
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses = torch.cat(
                (
                    train_losses,
                    torch.tensor([train_loss], dtype=torch.float64, device=self.device),
                )
            )
            self.scheduler.step()
            if (epoch + 1) % print_eval_rate == 0:
                self.eval()
                test_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs, targets = (
                            inputs.float().to(self.device),
                            targets.float().to(self.device),
                        )
                        outputs = self(inputs)
                        loss = loss_fn(outputs, targets)
                        test_loss += loss.item()

                test_loss /= len(test_loader)
                test_losses = torch.cat(
                    (
                        test_losses,
                        torch.tensor(
                            [test_loss], dtype=torch.float64, device=self.device
                        ),
                    )
                )
                log_message = f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Test loss: {test_loss:.4f}"
                # adapt the log message to use scientific notation when the value is too small
                if train_loss < 1e-3:
                    log_message = f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.2e} - Test loss: {test_loss:.2e}"
                pprint(0, log_message)

        if save_logs_path is not None:
            train_losses_np = train_losses.cpu().numpy()
            test_losses_np = test_losses.cpu().numpy()

            np.save(
                save_logs_path + "/train_losses_" + self.model_name + ".npy",
                train_losses_np,
            )
            np.save(
                save_logs_path + "/test_losses_" + self.model_name + ".npy",
                test_losses_np,
            )

        return {"train_loss": train_losses, "test_loss": test_losses}

    @cr("KAN.predict")
    def predict(
        self,
        X,
        return_targets: bool = False,
        **kwargs,
    ):
        r"""
        Predict the target values for the input data. The dataset is loaded to a DataLoader with the provided keyword arguments.
        The model is set to evaluation mode and the predictions are made using the input data. The output can be rescaled using
        the dataset scaler.

        Args:
            X: The dataset whose target values are to be predicted using the input data.
            rescale_output (bool): Whether to rescale the output with the scaler of the dataset (default: ``True``).
            kwargs (dict, optional): Additional keyword arguments to pass to the DataLoader. Can be used to set the parameters of the DataLoader (see PyTorch documentation at https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader):
                - batch_size (int, optional): Batch size (default: ``32``).
                - shuffle (bool, optional): Shuffle the data (default: ``True``).
                - num_workers (int, optional): Number of workers to use (default: ``0``).
                - pin_memory (bool, optional): Pin memory (default: ``True``).

        Returns:
            Tuple [np.ndarray, np.ndarray]: The predictions and the true target values.
        """
        dataloader_params = {
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0,
            "pin_memory": True,
        }

        for key in dataloader_params.keys():
            if key in kwargs:
                dataloader_params[key] = kwargs[key]

        predict_dataloader = DataLoader(X, **dataloader_params)

        total_rows = len(predict_dataloader.dataset)
        num_columns = self.output_size
        all_predictions = np.empty((total_rows, num_columns))
        all_targets = np.empty((total_rows, num_columns))

        self.eval()
        start_idx = 0
        with torch.no_grad():
            for x, y in predict_dataloader:
                output = self(x.to(self.device))
                batch_size = x.size(0)
                end_idx = start_idx + batch_size
                all_predictions[start_idx:end_idx, :] = output.cpu().numpy()
                all_targets[start_idx:end_idx, :] = y.cpu().numpy()
                start_idx = end_idx

        if return_targets:
            return all_predictions, all_targets
        else:
            return all_predictions

    def save(self, path: str, save_only_model: bool = False):
        r"""
        Save the model to a checkpoint file.

        Args:
            path (str): Path to save the model. It can be either a path to a directory or a file name.
            If it is a directory, the model will be saved with a filename that includes the number of epochs trained.
            save_only_model (bool, optional): Whether to only save the model, or also the optimizer and scheduler. Note that when this is true, you won't be able to resume training from checkpoint.(default: ``False``).
        """
        checkpoint = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden_size,
            "layer_type": self.layer_type,
            "model_name": self.model_name,
            "p_dropouts": self.p_dropouts,
            "state_dict": self.state_dict(),
        }
        if hasattr(self.input, "degree"):
            checkpoint["degree"] = self.input.degree

        if not save_only_model:
            checkpoint["optimizer"] = self.optimizer.state_dict()
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if os.path.isdir(path):
            filename = f"{self.model_name}.pth"
            path = path + filename

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")):
        """
        Loads a model from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
            device (torch.device): Device where the model is loaded (default: cpu).

        Returns:
            model (KAN): The loaded KAN model with the trained weights.
        """

        pprint(0, "Loading model...")
        checkpoint = torch.load(path, map_location=device)

        degree = checkpoint["degree"]
        layer_kwargs = {"degree": degree}

        model = cls(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            n_layers=checkpoint["n_layers"],
            hidden_size=checkpoint["hidden_size"],
            layer_type=checkpoint["layer_type"],
            model_name=checkpoint["model_name"],
            p_dropouts=checkpoint["p_dropouts"],
            device=device,
            **layer_kwargs,  # Pass the specific layer arguments
        )

        if 'optimizer' in checkpoint:
            model.optimizer_state_dict = checkpoint["optimizer"]
        if 'scheduler' in checkpoint:
            model.scheduler_state_dict = checkpoint["scheduler"]
            

        model.load_state_dict(checkpoint["state_dict"])
        pprint(0, f"Loaded KAN model: {checkpoint['model_name']}")
        keys_print = [
            "input_size",
            "output_size",
            "n_layers",
            "hidden_size",
            "layer_type",
            "p_dropouts",
        ]
        for key in keys_print:
            pprint(0, f"{key}: {checkpoint[key]}")

        return model


class ChebyshevLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Initialize Chebyshev polynomial tensors
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            cheby[:, :, 1] = x
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = (
                2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()
            )
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", cheby, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class JacobiLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.jacobi_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )

        nn.init.normal_(
            self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (
                (2 * i + self.a + self.b)
                * (2 * i + self.a + self.b - 1)
                / (2 * i * (i + self.a + self.b))
            )
            theta_k1 = (
                (2 * i + self.a + self.b - 1)
                * (self.a * self.a - self.b * self.b)
                / (2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            theta_k2 = (
                (i + self.a - 1)
                * (i + self.b - 1)
                * (2 * i + self.a + self.b)
                / (i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            )
            jacobi[:, :, i] = (
                (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone()
                - theta_k2 * jacobi[:, :, i - 2].clone()
            )  # 2 * x * jacobi[:, :, i - 1].clone() - jacobi[:, :, i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum(
            "bid,iod->bo", jacobi, self.jacobi_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
