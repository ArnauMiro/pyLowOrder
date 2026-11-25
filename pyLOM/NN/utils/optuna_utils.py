#!/usr/bin/env python
#
# pyLOM - Python Low Order Modeling.
#
# NN utility routines: seeds, devices and Optuna search spaces.
#
# Last rev: 23/11/2026

import os, random, time, numbers, numpy as np, torch

from typing import Optional, Union, Dict, Tuple, List, Any, Sequence
from torch  import Tensor

from ..              import DEVICE
from ...utils.errors import raiseError

def set_seed(seed: Optional[int] = None) -> None:
    """
    Set the random seed for full reproducibility across Python, NumPy, and PyTorch.
    
    This function enforces determinism in all major sources of randomness:
    - Python's built-in random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - cuBLAS/cuDNN kernels
    
    It also sets relevant environment variables to avoid hidden sources
    of non-determinism (e.g., Python hash seed, cuBLAS workspace).
    
    Note
    ----
    - Setting `torch.use_deterministic_algorithms(True)` may raise an error 
      if your code calls non-deterministic ops. In that case, rewrite those ops 
      or handle them explicitly.
    - Some performance optimizations (e.g., `torch.backends.cudnn.benchmark`) 
      are disabled to guarantee determinism.
    
    Args
    ----
    seed : int, optional
        The seed value to use for all RNGs. If ``None``, uses current UNIX time.
    """
    # -------------------------------
    # Python & NumPy RNGs
    # -------------------------------
    if seed is None:
        seed = int(time.time())

    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects the Python processes started afterwards (e.g. data loader workers)
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------------
    # PyTorch CPU & CUDA RNGs
    # -------------------------------
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # -------------------------------
    # cuBLAS/cuDNN Determinism
    # -------------------------------
    # cuBLAS (matrix multiplications) may introduce nondeterminism unless configured
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # or ":16:8" if OOM occurs

    try:
        # Enforce deterministic algorithms globally (PyTorch >= 1.8)
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        # Fallback for older versions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -------------------------------
    # Disable TF32 for Ampere+ GPUs
    # -------------------------------
    # TF32 can cause slight numerical differences across runs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _worker_init_fn(_):
    """Seed Python & NumPy per worker, derived from PyTorch worker seed.

    Notes
    -----
    - torch.initial_seed() already incorporates rank/worker id.
    - Use % 2**32 to obtain a numpy valid integer.
    - Do not use torch.manual_seed() here: Pytorch already sets each worker internally.
    """
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
 

def create_results_folder(RESUDIR: str, verbose: bool=True):
    r"""
    Create a folder to store the results of the neural network training.

    Args:
        RESUDIR (str): Path to the folder to be created.
        verbose (bool): If True, print messages to the console. Default is ``True``.
    """    
    if not os.path.exists(RESUDIR):
        os.makedirs(RESUDIR)
        if verbose: 
            print(f"Folder created: {RESUDIR}")
    elif verbose:
        print(f"Folder already exists: {RESUDIR}")
        

def select_device(device: str = DEVICE):
    r"""
    Select the device to be used for the training.

    Args:
        device (str): Device to be used. Default is cuda if available, otherwise cpu.
    """
    torch.device(device)
    return device


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The PyTorch model.
    Returns:
        int: The number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cleanup_tensors(tensors: Union[Tensor, Dict, None, Tuple, List]) -> None:
    r"""
    Drop the references to the tensors held by a dictionary or a list (their entries are set to ``None``) and empty the
    CUDA cache. A tensor or a tuple cannot be modified in place: only the cache is emptied (the caller must drop its
    own references).

    Args:
        tensors (Union[torch.Tensor, Dict, None, Tuple, List]): Tensors to release.
    """
    if tensors is None:
        return

    if isinstance(tensors, dict):
        for key in list(tensors.keys()):
            tensors[key] = None
    elif isinstance(tensors, list):
        for i in range(len(tensors)):
            tensors[i] = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_optimizing_value(name: str, spec: Any, trial):
    r"""
    Suggest a single hyperparameter value from a specification ``{"type": "int" | "float" | "categorical", ...}``
    (any other value is returned as it is, as a fixed value).

    Args:
        name (str): Name of the parameter in the Optuna trial.
        spec (Any): Specification of the parameter or fixed value.
        trial (optuna.Trial): Optuna trial.

    Returns:
        Any: Suggested (or fixed) value.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        return spec  # fixed value

    ptype = spec["type"]

    if ptype == "int":
        return trial.suggest_int(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step", 1),
            log=spec.get("log", False),
        )
    elif ptype == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    else:
        raise ValueError(f"Unsupported param type '{ptype}' for '{name}'")


def sample_params(space: dict, trial, prefix="") -> dict:
    """
    Recursively sample an Optuna search space dict.

    Args:
        space: Nested dict with fixed values or {type: ...} specs.
        trial: Optuna trial.
        prefix: Name prefix for hierarchical parameters.
    """
    result = {}
    for k, v in space.items():
        full_name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and "type" not in v:
            # Nested section → recurse
            result[k] = sample_params(v, trial, prefix=full_name)
        else:
            result[k] = get_optimizing_value(full_name, v, trial)
    return result

def _is_dist_spec(x: object) -> bool:
    return isinstance(x, dict) and "type" in x


def _materialize_space(space: dict, flat: dict, prefix: str = "") -> dict:
    r"""
    Replace the specifications ``{"type": ...}`` of ``space`` with the values of ``flat`` (dotted keys), keeping the
    fixed values and going through the nested dictionaries recursively.
    """
    out = {}
    for k, v in space.items():
        path = f"{prefix}.{k}" if prefix else k
        if _is_dist_spec(v):
            if path not in flat:
                raise ValueError(f"Missing sampled value for '{path}'")
            out[k] = flat[path]
        elif isinstance(v, dict):
            out[k] = _materialize_space(v, flat, path)
        else:
            out[k] = flat.get(path, v)
    return out


def suggest_params(optimization_params: Dict, trial, fixed: Sequence[str] = ()) -> Dict:
    r"""
    Sample the parameters of a trial from the search space of ``create_optimized_model`` (MLP, KAN):

    - A pair of numbers ``(low, high)`` is a range of integers or floats, sampled in log scale if it spans three orders of magnitude or more.
      A pair of powers of two more than one power apart is sampled as a list of powers of two (categorical: use ``"pow2"`` below for an ordered value).
    - Any other tuple or list is a list of choices. Choices that are not plain values, such as layer classes or activation functions, are
      recorded in Optuna by name.
    - A dictionary with a ``"type"`` key is an explicit specification, and any other dictionary is searched recursively (parameter
      ``<name>.<key>``):

      - ``{"type": "int" | "float" | "categorical", ...}``: see ``get_optimizing_value``.
      - ``{"type": "pow2", "low": 4, "high": 256}`` is a power of two between ``low`` and ``high``: Optuna samples its exponent as an integer
        (parameter ``<name>_log2``), so that samplers such as QMC and TPE see an ordered value.
      - ``{"type": "towards_one", "low": 0.9, "high": 0.9996}`` is a value between ``low`` and ``high``, below 1, such as the ``gamma`` of a
        scheduler: Optuna samples ``1 - value`` in log scale (parameter ``<name>_one_minus``), so that the values get denser towards 1.
      - ``{"type": "batch_scaled", "low": 3e-4, "high": 1e-2, "reference": 512, "exponent": 0.5}`` (top-level parameters only, such as
        ``lr``) is a value that grows with the batch size of the trial: ``value = base * (batch_size / reference) ** exponent``, where Optuna
        samples ``base`` between ``low`` and ``high`` in log scale (parameter ``<name>_ref``, the value at the reference batch size).
        The search space stays fixed for the samplers (``base`` and the batch size are independent parameters, as QMC needs), but each batch
        size searches its own range of values. ``exponent`` is 0.5 by default (square root scaling of the learning rate, the usual one for
        Adam), and ``"batch_key"`` is the parameter of the batch size (``"batch_size"`` by default), sampled or fixed. The value is stored in
        the user attribute ``derived_params`` of the trial.

    - Any other value is fixed.

    Args:
        optimization_params (Dict): The search space.
        trial (optuna.Trial): The Optuna trial (an ``optuna.trial.FixedTrial`` gives the parameters of a trial already run, see ``trial_params``).
        fixed (Sequence[str]): Top-level parameters that are always fixed values, even if they look like a range (e.g. the
            ``input_domain`` of a KAN, a ``(low, high)`` pair) (default: ``()``).

    Returns:
        Dict: The parameters of the trial, with the same keys as ``optimization_params``.
    """
    # The batch_scaled values depend on the batch size of the trial: they are suggested after the other parameters
    params = {key: space if key in fixed else _suggest_value(key, space, trial)
              for key, space in optimization_params.items() if not _is_batch_scaled(space)}
    for key, space in optimization_params.items():
        if _is_batch_scaled(space):
            params[key] = _suggest_batch_scaled(key, space, params, trial)
    return {key: params[key] for key in optimization_params}


def trial_params(optimization_params: Dict, params: Dict, fixed: Sequence[str] = ()) -> Dict:
    r"""
    Parameters of a trial of ``create_optimized_model``: the search space with the values that Optuna sampled in that trial (see
    ``suggest_params``). Useful to train again any trial of a study, not only the best one.

    Args:
        optimization_params (Dict): The search space given to ``create_optimized_model``.
        params (Dict): The parameters of the trial as stored by Optuna, e.g. ``study.trials[i].params`` (flat names, such as
            ``scheduler_kwargs.gamma`` or ``batch_size_log2``).
        fixed (Sequence[str]): Top-level parameters that are always fixed values (see ``suggest_params``; default: ``()``).

    Returns:
        Dict: The parameters, to build the model and to train it.
    """
    from optuna.trial import FixedTrial
    return suggest_params(optimization_params, FixedTrial(params), fixed=fixed)


def _is_batch_scaled(space):
    return isinstance(space, dict) and space.get("type") == "batch_scaled"


def _suggest_batch_scaled(name, space, params, trial):
    # value = base * (batch_size / reference) ** exponent, with base sampled in log scale: the search space of the
    # samplers is fixed (base and the batch size are independent), but the range of the value moves with the batch size
    batch_key = space.get("batch_key", "batch_size")
    if batch_key not in params:
        raiseError(f"'{name}' is scaled with the batch size '{batch_key}', which is not a parameter")
    low, high, reference = space["low"], space["high"], space["reference"]
    exponent = space.get("exponent", 0.5)
    if not 0 < low < high or not reference > 0:
        raiseError(f"The bounds of '{name}' must satisfy 0 < low < high and reference > 0, got {low}, {high} and {reference}")
    value = trial.suggest_float(f"{name}_ref", low, high, log=True) * (params[batch_key] / reference) ** exponent
    # Optuna only stores base: the value is kept as a user attribute of the trial
    trial.set_user_attr("derived_params", {**trial.user_attrs.get("derived_params", {}), name: value})
    return value


def _suggest_value(name, space, trial):
    if isinstance(space, dict):
        if _is_batch_scaled(space):
            raiseError(f"'{name}': batch_scaled is only supported for top-level parameters")
        if space.get("type") == "pow2":
            # Sampled through its exponent, so that the samplers see an ordered integer
            low, high = space["low"], space["high"]
            if not (_is_power_of_2(low) and _is_power_of_2(high)):
                raiseError(f"The bounds of '{name}' must be powers of two, got {low} and {high}")
            return 2 ** trial.suggest_int(f"{name}_log2", int(low).bit_length() - 1, int(high).bit_length() - 1)
        if space.get("type") == "towards_one":
            # A value just below 1 (e.g. the gamma of a scheduler): 1 - value is sampled in log scale, as an ordered float
            low, high = space["low"], space["high"]
            if not 0 < low < high < 1:
                raiseError(f"The bounds of '{name}' must satisfy 0 < low < high < 1, got {low} and {high}")
            return 1.0 - trial.suggest_float(f"{name}_one_minus", 1.0 - high, 1.0 - low, log=True)
        if "type" in space:
            return get_optimizing_value(name, space, trial)
        suggested_dict = {}
        for key, subspace in space.items():
            full_name = f"{name}.{key}"
            suggested_dict[key] = _suggest_value(full_name, subspace, trial)
        return suggested_dict

    if isinstance(space, (tuple, list)):
        if _is_range(space):
            low, high = space
            if isinstance(low, numbers.Integral) and isinstance(high, numbers.Integral):
                low, high = int(low), int(high)

                if _is_power_of_2(low) and _is_power_of_2(high):
                    power_low = int(np.log2(low))
                    power_high = int(np.log2(high))
                    power_diff = power_high - power_low

                    if power_diff > 1:
                        choices = [2**p for p in range(power_low, power_high + 1)]
                        return trial.suggest_categorical(name, choices)

                return trial.suggest_int(name, low, high, log=_log_scale(low, high))

            low, high = float(low), float(high)
            return trial.suggest_float(name, low, high, log=_log_scale(low, high))

        if all(_is_plain(choice) for choice in space):
            return trial.suggest_categorical(name, list(space))
        # Optuna storages only keep plain values: other choices are suggested by name
        names = [_choice_name(choice) for choice in space]
        if len(set(names)) != len(names):
            raiseError(f"The choices of '{name}' must have different names, got: {names}")
        return space[names.index(trial.suggest_categorical(name, names))]

    return space


def _flatten_params(params, prefix=""):
    # Nested parameters back to the flat names of Optuna ("scheduler_kwargs.gamma")
    flat = {}
    for key, value in params.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, name))
        else:
            flat[name] = value
    return flat


def _is_power_of_2(n):
    return isinstance(n, numbers.Integral) and n > 0 and (n & (n - 1)) == 0


def _is_plain(value):
    return value is None or isinstance(value, (bool, int, float, str))


def _is_range(space):
    return len(space) == 2 and all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in space)


def _log_scale(low, high):
    return low > 0 and high / low >= 1e3 * (1 - 1e-9)


def _choice_name(value):
    return getattr(value, "__name__", None) or repr(value)


