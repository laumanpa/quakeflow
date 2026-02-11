"""
Helper functions for QuakeFlow.
"""

from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any
from scipy.signal import hilbert


def compute_amplitude(data: np.ndarray,
                      sampling_rate: float,
                      method: str = "max",
                      percentile: float = 95.0) -> float:
    """Compute robust amplitude from windowed data.

    Methods:
    - 'max': max absolute amplitude
    - 'rms': root-mean-square
    - 'percentile': percentile of absolute amplitude (e.g., 95)
    - 'envelope': peak of Hilbert envelope (optionally smoothed outside)
    """
    if data is None or len(data) == 0:
        return float('nan')

    x = np.asarray(data, dtype=np.float32)
    if not np.isfinite(x).any():
        return float('nan')

    if method == "rms":
        return float(np.sqrt(np.mean(x ** 2)))
    elif method == "percentile":
        return float(np.percentile(np.abs(x), percentile))
    elif method == "envelope":
        env = np.abs(hilbert(x))
        return float(np.max(env))
    else:  # default 'max'
        return float(np.max(np.abs(x)))


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_keys = {
        'paths.base_dir',
        'stations.station_code',
        'template_matching.start_date',
        'template_matching.days_to_process'
    }
    
    missing = []
    for key in required_keys:
        keys = key.split('.')
        value = config
        for k in keys:
            if k not in value:
                missing.append(key)
                break
            value = value[k]
    
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    return True


def format_timedelta(delta: timedelta) -> str:
    """Format timedelta to human-readable string."""
    total_seconds = int(delta.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:
        parts.append(f"{seconds}s")
    
    return " ".join(parts)


def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """Calculate basic statistics of data."""
    if len(data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan
        }
    
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data))
    }


def time_range_to_dates(start_date: str, days: int) -> pd.DatetimeIndex:
    """Convert start date and days to date range."""
    return pd.date_range(
        start=start_date,
        periods=days,
        freq="D"
    )


def save_dataframe(df: pd.DataFrame, path: Path, **kwargs):
    """Save DataFrame with consistent settings."""
    default_kwargs = {
        'index': False,
        'float_format': '%.6e'
    }
    default_kwargs.update(kwargs)
    df.to_csv(path, **default_kwargs)