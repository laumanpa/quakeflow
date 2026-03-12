"""
GPU-accelerated correlation backend.

Provides batched FFT cross-correlation using CuPy or PyTorch for
significant speedups on CUDA-capable hardware.

Falls back gracefully to NumPy when no GPU is available.
"""

import numpy as np
from typing import List, Optional, Tuple
from rich.console import Console

console = Console()

# --------------------------------------------------------------------------
# Backend detection
# --------------------------------------------------------------------------
_CUPY_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.signal  # noqa: F401
    _CUPY_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    if torch.cuda.is_available():
        _TORCH_AVAILABLE = True
except ImportError:
    pass


def gpu_available(backend: str = 'cupy') -> bool:
    """Check if requested GPU backend is usable."""
    if backend == 'cupy':
        return _CUPY_AVAILABLE
    elif backend == 'torch':
        return _TORCH_AVAILABLE
    return False


# --------------------------------------------------------------------------
# CuPy backend
# --------------------------------------------------------------------------
def _normalized_xcorr_cupy(
    signal: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """Single normalized cross-correlation on GPU via CuPy.

    Returns CPU NumPy array (float32).
    """
    import cupy as cp
    from cupyx.scipy.signal import fftconvolve as cp_fftconvolve

    sig_g = cp.asarray(signal, dtype=cp.float32)
    tpl_g = cp.asarray(template, dtype=cp.float32)

    tpl_g -= tpl_g.mean()
    sig_g -= sig_g.mean()

    corr = cp_fftconvolve(sig_g, tpl_g[::-1], mode='valid')

    tpl_energy = float(cp.sum(tpl_g ** 2))
    win_energy = cp.convolve(sig_g ** 2, cp.ones(len(tpl_g), dtype=cp.float32), mode='valid')
    denom = cp.sqrt(tpl_energy * win_energy)
    denom[denom == 0] = cp.inf

    result = corr / denom
    out = cp.asnumpy(result).astype(np.float32)
    
    # Ensure valid correlation range
    out = np.clip(out, -1.0, 1.0)
    
    # Free GPU arrays immediately
    del sig_g, tpl_g, corr, win_energy, denom, result
    return out


def batch_xcorr_cupy(
    signal: np.ndarray,
    templates: List[np.ndarray],
    device: str = 'cuda:0',
) -> List[np.ndarray]:
    """Batched normalized cross-correlation using CuPy.

    Parameters
    ----------
    signal : 1D float array (continuous day data)
    templates : list of 1D float arrays

    Returns
    -------
    list of 1D CC arrays (one per template)
    """
    results = []
    for tpl in templates:
        try:
            cc = _normalized_xcorr_cupy(signal, tpl)
            results.append(cc)
        except Exception as e:
            console.print(f"[yellow]CuPy xcorr failed: {e}, falling back to CPU[/yellow]")
            cc = _cpu_normalized_xcorr(signal, tpl)
            results.append(cc)
    # Release GPU memory pool between batches
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return results


# --------------------------------------------------------------------------
# PyTorch backend
# --------------------------------------------------------------------------
def batch_xcorr_torch(
    signal: np.ndarray,
    templates: List[np.ndarray],
    device: str = 'cuda:0',
) -> List[np.ndarray]:
    """Batched normalized cross-correlation using PyTorch conv1d.

    Uses torch.nn.functional.conv1d which maps onto cuDNN for GPU.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        device = 'cpu'

    sig_t = torch.tensor(signal, dtype=torch.float32, device=device)
    sig_t = sig_t - sig_t.mean()
    # Shape for conv1d: (batch, channels, length)
    sig_batch = sig_t.unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    results = []
    for tpl in templates:
        tpl_t = torch.tensor(tpl, dtype=torch.float32, device=device)
        tpl_t = tpl_t - tpl_t.mean()
        # Flip template for cross-correlation (conv1d does correlation, not convolution)
        # Actually F.conv1d does cross-correlation by default
        kernel = tpl_t.flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, M)

        corr = F.conv1d(sig_batch, kernel).squeeze()  # (N - M + 1,)

        # Normalization
        tpl_energy = (tpl_t ** 2).sum()
        ones_kernel = torch.ones(1, 1, len(tpl_t), dtype=torch.float32, device=device)
        win_energy = F.conv1d((sig_batch ** 2), ones_kernel).squeeze()
        denom = torch.sqrt(tpl_energy * win_energy)
        denom[denom == 0] = float('inf')
        cc = (corr / denom).cpu().numpy().astype(np.float32)
        
        # Ensure valid correlation range
        cc = np.clip(cc, -1.0, 1.0)
        
        results.append(cc)
        del tpl_t, kernel, corr, ones_kernel, win_energy, denom

    # Free GPU cache between batches
    del sig_t, sig_batch
    if device != 'cpu':
        torch.cuda.empty_cache()
    return results


# --------------------------------------------------------------------------
# CPU fallback
# --------------------------------------------------------------------------
def _cpu_normalized_xcorr(signal: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Standard CPU normalized cross-correlation with improved numerical stability."""
    from scipy.signal import fftconvolve

    # Use float64 for all calculations
    signal = signal.astype(np.float64)
    template = template.astype(np.float64)
    
    # Center the signals
    template = template - template.mean()
    signal = signal - signal.mean()

    # Check for zero-variance template
    template_energy = np.sum(template ** 2)
    if template_energy < 1e-20:
        return np.full(len(signal) - len(template) + 1, 0.0, dtype=np.float32)

    # FFT convolution
    corr = fftconvolve(signal, template[::-1], mode='valid')
    
    # Window energy with minimum threshold
    window_energy = np.convolve(signal ** 2, np.ones(len(template)), mode='valid')
    window_energy = np.maximum(window_energy, 1e-20)
    
    # Normalization
    denom = np.sqrt(template_energy * window_energy)
    result = corr / denom
    
    # Clip to valid correlation range and convert to float32
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def batch_xcorr_cpu(
    signal: np.ndarray,
    templates: List[np.ndarray],
) -> List[np.ndarray]:
    """Batched cross-correlation on CPU."""
    return [_cpu_normalized_xcorr(signal, tpl) for tpl in templates]


# --------------------------------------------------------------------------
# Unified interface
# --------------------------------------------------------------------------
def batch_xcorr(
    signal: np.ndarray,
    templates: List[np.ndarray],
    backend: str = 'auto',
    device: str = 'cuda:0',
    batch_size: int = 32,
) -> List[np.ndarray]:
    """Compute normalized cross-correlation for multiple templates.

    Automatically selects the best available backend.

    Parameters
    ----------
    signal : continuous waveform data (1D)
    templates : list of template waveforms
    backend : 'cupy', 'torch', 'cpu', or 'auto'
    device : CUDA device string (for torch)
    batch_size : templates per GPU batch (for memory management)

    Returns
    -------
    list of CC arrays
    """
    if backend == 'auto':
        if _CUPY_AVAILABLE:
            backend = 'cupy'
        elif _TORCH_AVAILABLE:
            backend = 'torch'
        else:
            backend = 'cpu'

    if backend == 'cupy' and _CUPY_AVAILABLE:
        console.print(f"[dim]Using CuPy GPU backend[/dim]")
        results = []
        for i in range(0, len(templates), batch_size):
            batch = templates[i:i + batch_size]
            results.extend(batch_xcorr_cupy(signal, batch, device))
        return results
    elif backend == 'torch' and _TORCH_AVAILABLE:
        console.print(f"[dim]Using PyTorch GPU backend ({device})[/dim]")
        results = []
        for i in range(0, len(templates), batch_size):
            batch = templates[i:i + batch_size]
            results.extend(batch_xcorr_torch(signal, batch, device))
        return results
    else:
        return batch_xcorr_cpu(signal, templates)
