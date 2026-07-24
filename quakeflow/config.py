"""
Configuration management for QuakeFlow.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import copy
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
            'lon': 7.9021,
            # Multi-station / multi-component settings
            'networks': [],  # list of {code, network, channels, lat, lon, weight}
            'use_all_components': False,  # correlate on all channels and stack
            'stacking_method': 'mean',  # mean, median, pws (phase-weighted stack)
            'component_weights': {'Z': 1.0, 'N': 0.5, 'E': 0.5},
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
            'amplitude_percentile': 95.0,
            'magnitude_column': None,  # generic-catalog magnitude column name (None = auto-detect)
            'noise_window': None,  # defaults to pre_event at runtime
            'snuffler_markers_file': None,  # path to Pyrocko Snuffler markers file
            'marker_match_tolerance': 120.0,  # seconds tolerance for marker matching
            'velocity_model': None,  # path to 1D velocity model (.nd/.tvel) for P travel-time estimation; None = ak135
        },
        'template_matching': {
            'min_snr': 2.0,
            'similarity_threshold': 0.5,
            'distance_samples': 200,
            'min_spike_ratio': 15.0,
            'cluster_eps': 0.2,
            'cluster_enabled': True,
            'n_jobs': 4,
            'start_date': '2018-01-01',
            'days_to_process': 365,
            'pre_amplitude': 0.5,
            'post_amplitude': 6.0,
            'amplitude_method': 'max',
            'amplitude_percentile': 95.0,
            'pre_event': 0.5,
            'post_event': 5.0,
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
            'calibration_method': 'isotonic',
            'compute_spectral': False,
            'compute_station_corrections': True,
            'compute_uncertainty': True,
            'compute_all_amplitude_methods': False,
            'plot_waveform_comparisons': True,
        },
        'detection_qc': {
            'compute_snr': True,
            'snr_noise_window': 2.0,  # seconds before detection for noise
            'snr_signal_window': 2.0,  # seconds after detection for signal
            'compute_cc_sharpness': True,
            'cc_sharpness_window': 20,  # samples around peak for sharpness
            'min_detection_snr': 0.0,  # minimum SNR to keep a detection
            'edge_margin_seconds': 120.0,  # skip detections within this many seconds of day start/end
        },
        'relocation': {
            'enabled': False,
            'output_format': 'hypodd',  # hypodd or growclust
            'max_dt_cc_pairs': 50,  # max pairs per event
            'min_cc_for_dt': 0.5,  # minimum CC for dt measurement
            'velocity_model': None,  # path to 1D velocity model
        },
        'template_updating': {
            'enabled': False,
            'min_similarity': 0.8,
            'min_snr': 5.0,
            'max_templates': 500,
            'update_interval_days': 30,
        },
        'realtime': {
            'enabled': False,
            'state_file': 'quakeflow_state.json',
            'chunk_hours': 1,
            'alert_threshold': 0.8,
            'alert_callback': None,  # callable or endpoint URL
        },
        'gpu': {
            'enabled': False,
            'backend': 'cupy',  # cupy or torch
            'batch_size': 32,
            'device': 'cuda:0',
        },
        'squirrel': {
            'persistent': 'eifel3'
        },
        'sds': {
            'root': None,             # path to SDS archive root; set to enable SDS backend
            'type': 'D',              # SDS data type character (D = waveform data)
            'cache_size': 64,         # max day-files kept in LRU cache
            'max_workers': 8,         # thread pool size for parallel reads
            'fileborder_seconds': 30.0,  # extra seconds at day boundaries
        },
        'qseek_filter': {
            'min_semblance': 0.3,            # minimum semblance value
            'min_n_picks': 6,                # minimum number of phase picks
            'max_uncertainty_horizontal': 500.0,  # max horizontal location uncertainty (m)
        },
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration from file or defaults."""
        # Deep copy to avoid mutating class-level DEFAULT_CONFIG
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                if loaded_config:
                    self._deep_update(self.config, loaded_config)

            # Support legacy/mis-indented configs where 'networks' may be
            # placed at top-level instead of under the 'stations' section.
            # If a top-level 'networks' is present, move it into
            # self.config['stations']['networks'] so downstream code picks it up.
            try:
                if 'networks' in self.config and isinstance(self.config.get('networks'), list):
                    # Only move when stations.networks is empty/default
                    st = self.config.get('stations', {})
                    if not st.get('networks'):
                        self.config['stations']['networks'] = self.config.pop('networks')
            except Exception:
                pass

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
            # Deep copy to avoid mutating self.config when converting Paths
            config_to_save = copy.deepcopy(self.config)
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