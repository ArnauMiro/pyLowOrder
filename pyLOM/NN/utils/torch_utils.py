import torch
from torch import nn, Tensor

def snap(x: torch.Tensor, name: str):
    """Lightweight tensor snapshot to avoid heavy string formatting."""
    x32 = x.detach().to(torch.float32, copy=False)
    finite = bool(torch.isfinite(x32).all())
    print(f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}, "
          f"contiguous={x.is_contiguous()}, finite={finite}, "
          f"mean={x32.mean().item():.6f}, std={x32.std(unbiased=False).item():.6f}, "
          f"min={x32.min().item():.6f}, max={x32.max().item():.6f}")
    print(f"{name} sample:", x32.flatten()[:8])

def assert_finite(x: torch.Tensor, name: str):
    """Raise early if non-finite values appear."""
    if not torch.isfinite(x).all():
        raise RuntimeError(f"Non-finite values in {name}")

def ensure_2d_f32_contig(x: torch.Tensor) -> torch.Tensor:
    """Normalize layout for Linear: 2D, float32, contiguous."""
    if x.dim() != 2:
        x = x.view(x.size(0), -1)
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x

def safe_linear(
    x: Tensor,
    layer: nn.Linear,
    *,
    chunk: int = 64,
    compute_dtype: torch.dtype = torch.float64,
    single_thread_mm: bool = True,
    debug_name: str = ""
) -> Tensor:
    """
    Ultra-stable Linear path for CPU:
    - Forces contiguous transposed weights (Wt = W^T).
    - Performs explicit mm in small chunks to avoid large-GEMM edge cases.
    - Optionally runs each mm with a single thread to avoid OMP/MKL races.
    - Computes in `compute_dtype` and casts back to input dtype.

    Args:
        x: [N, in_features] input.
        layer: nn.Linear with weight [out_features, in_features] and optional bias.
        chunk: Row chunk size for blocked mm.
        compute_dtype: Internal compute dtype (float64 default for robustness).
        single_thread_mm: If True, temporarily set torch num_threads=1 for each mm.
        debug_name: Optional tag to identify which layer is calling.

    Returns:
        y: [N, out_features] tensor.
    """
    assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"
    orig_dtype = x.dtype

    # Normalize layout/dtypes
    x = x.to(dtype=compute_dtype, copy=False)
    W = layer.weight.to(dtype=compute_dtype)
    b = layer.bias.to(dtype=compute_dtype) if layer.bias is not None else None

    # Contiguous RHS for mm
    Wt = W.t().contiguous()  # [in_features, out_features]

    N = x.size(0)
    out_features = W.size(0)
    y = x.new_empty((N, out_features), dtype=compute_dtype)

    # Prepare optional single-thread block
    prev_threads = torch.get_num_threads() if single_thread_mm else None

    # Blocked matmul
    for i in range(0, N, chunk):
        xi = x[i:i + chunk].contiguous()  # [chunk, in_features]
        if single_thread_mm:
            torch.set_num_threads(1)
        try:
            yi = torch.mm(xi, Wt)         # [chunk, out_features]
        finally:
            if single_thread_mm:
                torch.set_num_threads(prev_threads)
        if b is not None:
            yi.add_(b)
        y[i:i + chunk] = yi
        # (Opcional) trazas ligeras por bloque en depuración:
        # print(f"{debug_name} mm chunk {i}:{min(i+chunk, N)}")

    # Cast back
    if y.dtype != orig_dtype:
        y = y.to(orig_dtype)
    return y


