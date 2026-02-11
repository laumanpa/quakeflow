"""
Configuration management for QuakeFlow.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class Config:
    """Configuration manager for the pipeline."""
    
    DEFAULT_CONFIG = {
        'paths': {
            'base_dir': './template_matching',
            'templates_dir': './templates',
            'output_dir': './results',
            'plots_dir': './plots',
            'catalog_file': None,
            'template_info_file': 'template_info.csv'
        },
        'stations': {
            'station_code': 'GWBD',
            'channels': ['Z', 'N', 'E'],
            'primary_channel': 'Z',
            'lat': 50.114,
            'lon': 7.9021
        },
        'template_creation': {
            'pre_event': 0.5,
            'post_event': 5.0,
            'sta_window': 0.5,
            'lta_window': 5.0,
            'onset_thr_on': 3.5,
            'onset_thr_off': 1.0,
            'filter_min': 1.0,
            'filter_max': 30.0,
            'amplitude_method': 'max',
            'amplitude_percentile': 95.0
        },
        'template_matching': {
            'min_snr': 2.0,
            'similarity_threshold': 0.5,
            'distance_samples': 200,
            'min_spike_ratio': 3.0,
            'cluster_eps': 0.2,
            'n_jobs': 4,
            'start_date': '2018-01-01',
            'days_to_process': 365,
            'pre_amplitude': 0.5,
            'post_amplitude': 6.0,
            'amplitude_method': 'max',
            'amplitude_percentile': 95.0,
            'domain': 'fft',
            'wavelet': {
                'num_scales': 12,
                'min_period': 0.05,
                'max_period': 0.5,
                'wavelet_w': 6.0,
                'scale_weighting': 'uniform',
                'decimate_factor': 1,
                'reuse_signal_cwt': True
            },
            'wst': {
                'window_seconds': 5.5,
                'hop_seconds': 0.1,
                'J': 6,
                'Q': 8,
                'max_order': 2,
                'metric': 'cosine',
                'similarity_threshold': None,
                'pad_mode': 'zero',
                'backend': 'numpy',
                'device': 'cpu'
            }
        },
        'evaluation': {
            'reference_distance': 1.0,
            'attenuation_factor': 1.0,
            'min_magnitude': 0.0,
            'mc_method': 'maxcurvature',
            'geometrical_spreading': 1.0,
            'calibration_method': 'isotonic'
        },
        'squirrel': {
            'persistent': 'eifel3'
        }
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from file or defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self._deep_update(self.config, loaded_config)
        
        self._convert_paths()
    
    def _deep_update(self, original: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _convert_paths(self):
        """Convert string paths to Path objects."""
        for key in self.config['paths']:
            if key.endswith('_dir') and self.config['paths'][key]:
                self.config['paths'][key] = Path(self.config['paths'][key])
            elif key.endswith('_file') and self.config['paths'][key]:
                self.config['paths'][key] = Path(self.config['paths'][key])
    
    def save(self, path: Path):
        """Save configuration to file."""
        with open(path, 'w') as f:
            # Convert Path objects to strings for YAML
            config_to_save = self.config.copy()
            for key in config_to_save['paths']:
                if isinstance(config_to_save['paths'][key], Path):
                    config_to_save['paths'][key] = str(config_to_save['paths'][key])
            yaml.dump(config_to_save, f, default_flow_style=False)
    
    def get_path(self, key: str) -> Path:
        """Get a path and ensure its parent exists."""
        path = self.config['paths'][key]
        if isinstance(path, Path) and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def __getitem__(self, key: str) -> Any:
        """Access configuration using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                raise KeyError(f"Config key '{key}' not found")
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a configuration value."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def update(self, key: str, value: Any):
        """Update a configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            config = config[k]
        
        config[keys[-1]] = value