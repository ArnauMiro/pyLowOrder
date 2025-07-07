from typing import Union
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

class ShapeValidator:
    """
    Input validator for the GNS model.

    This class encapsulates input validation logic for GNS, ensuring that inputs to
    `predict`, `fit`, or other routines conform to the expected shapes and dimensions.

    Supports both raw tensor inputs and datasets containing batched inputs and targets.

    Args:
        input_dim (int): Expected dimensionality of each input vector (D).
        output_dim (int, optional): Expected number of output features (F) per node. Used for target validation.
    """

    def __init__(self, input_dim: int, output_dim: int = None) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

    def validate(self, X: Union[Tensor, TorchDataset]) -> None:
        """
        Validate the input type and shape.

        Args:
            X (Tensor or TorchDataset): Input to validate. Either:
                - Tensor of shape [D]
                - Dataset yielding (x, y) pairs with shapes:
                    - x: [D]
                    - y: [N, F] if `output_dim` is specified

        Raises:
            ValueError or TypeError if the input is malformed.
        """
        if isinstance(X, Tensor):
            self._validate_tensor(X)
        elif isinstance(X, TorchDataset):
            self._validate_dataset(X)
        else:
            raise TypeError("Input must be a Tensor or a TorchDataset")

    def _validate_tensor(self, x: Tensor) -> None:
        if x.ndim != 1:
            raise ValueError(f"Expected input Tensor of shape [B, D], got {x.shape}")
        if x.shape[0] != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")

    def _validate_dataset(self, dataset: TorchDataset) -> None:
        sample = dataset[0]
        if isinstance(sample, (tuple, list)):
            x_sample, y_sample = sample
        else:
            x_sample, y_sample = sample, None

        if x_sample.ndim != 1:
            raise ValueError(f"Expected input sample of shape [D], got {x_sample.shape}")
        if x_sample.shape[0] != self.input_dim:
            raise ValueError(f"Input sample dimension mismatch: expected {self.input_dim}, got {x_sample.shape[0]}")

        if y_sample is not None and self.output_dim is not None:
            if y_sample.ndim != 2:
                raise ValueError(f"Expected target sample of shape [N, F], got {y_sample.shape}")
            if y_sample.shape[-1] != self.output_dim:
                raise ValueError(f"Target output dim mismatch: expected {self.output_dim}, got {y_sample.shape[-1]}")