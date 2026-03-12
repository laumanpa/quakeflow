"""
Template matching on continuous data.

Supports:
- Single or multi-station correlation stacking
- Multi-component (Z/N/E) correlation stacking
- FFT, wavelet, and WST correlation domains
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from obspy import Stream, UTCDateTime, read
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from scipy.signal import fftconvolve, find_peaks

from .base import BaseProcessor
from ..config import Config
from ..utils.helpers import compute_amplitude
import yaml
from ..utils.plotting import plot_templates_tsne


console = Console()


class TemplateMatcher(BaseProcessor):
    """Perform template matching on continuous data."""
    
    # Maximum entries in the scattering transform cache to bound memory
    _MAX_SCATTERING_CACHE = 4

    def __init__(self, config: Config):
        super().__init__(config)
        self.MIN_TOTAL_NPTS = 500
        self._scattering_cache = {}

    # ------------------------------------------------------------------
    # CC stacking for multi-station / multi-component
    # ------------------------------------------------------------------
    @staticmethod
    def _stack_cc_traces(cc_list: List[np.ndarray],
                         weights: Optional[List[float]] = None,
                         method: str = 'mean') -> Optional[np.ndarray]:
        """Stack multiple normalised CC traces into a single one.

        Parameters
        ----------
        cc_list : list of 1-D arrays (all same length after trimming)
        weights : optional per-trace weights (station/component)
        method  : 'mean' | 'median' | 'pws' (phase-weighted stack)

        Returns
        -------
        1-D stacked CC array (float32), or None if cc_list is empty.
        """
        if not cc_list:
            return None

        # Trim all CCs to shortest length (small edge effects from different trace lengths)
        min_len = min(len(c) for c in cc_list)
        # Use float32 to reduce memory (CC values are in [-1, 1])
        cc_arr = np.column_stack([c[:min_len].astype(np.float32) for c in cc_list])  # (min_len, N)

        if weights is None:
            w = np.ones(cc_arr.shape[1], dtype=np.float32)
        else:
            w = np.asarray(weights[:cc_arr.shape[1]], dtype=np.float32)
        w /= w.sum() + 1e-30

        if method == 'median':
            result = np.median(cc_arr, axis=1).astype(np.float32)
        elif method == 'pws':
            # Phase-Weighted Stack (Schimmel & Paulssen 1997)
            # Process one trace at a time to avoid allocating full complex128 array
            from scipy.signal import hilbert as sp_hilbert
            phase_sum = np.zeros(min_len, dtype=np.complex64)
            for i in range(cc_arr.shape[1]):
                analytic_i = sp_hilbert(cc_arr[:, i]).astype(np.complex64)
                phase_sum += np.exp(1j * np.angle(analytic_i))
                del analytic_i
            coherence = (np.abs(phase_sum / cc_arr.shape[1]) ** 2).astype(np.float32)
            del phase_sum
            linear_stack = np.average(cc_arr, axis=1, weights=w).astype(np.float32)
            result = linear_stack * coherence
        else:
            # weighted mean (default)
            result = np.average(cc_arr, axis=1, weights=w).astype(np.float32)

        del cc_arr
        return result

    # ------------------------------------------------------------------
    # Detection QC metrics
    # ------------------------------------------------------------------
    def _compute_detection_qc(self, trace, detection: Dict, sr: float) -> Dict:
        """Compute QC metrics for a single detection.

        Adds:
        - detection_snr : signal-to-noise ratio around the detection
        - cc_sharpness  : how peaked the CC is around the detection (kurtosis-like)
        """
        qc = {}
        t = detection.get('time')
        qc_cfg = self.config.get('detection_qc', {})

        # --- SNR at detection ---
        if qc_cfg.get('compute_snr', True) and t is not None and trace is not None:
            noise_win = float(qc_cfg.get('snr_noise_window', 2.0))
            sig_win = float(qc_cfg.get('snr_signal_window', 2.0))
            try:
                noise_tr = trace.slice(t - noise_win, t)
                sig_tr = trace.slice(t, t + sig_win)
                noise_rms = np.sqrt(np.mean(noise_tr.data ** 2)) if len(noise_tr.data) > 0 else 0.0
                sig_rms = np.sqrt(np.mean(sig_tr.data ** 2)) if len(sig_tr.data) > 0 else 0.0
                qc['detection_snr'] = float(sig_rms / noise_rms) if noise_rms > 0 else np.nan
            except Exception:
                qc['detection_snr'] = np.nan
        else:
            qc['detection_snr'] = np.nan

        # --- CC peak sharpness ---
        # Prefer precomputed '_cc_sharpness' (set during peak detection to avoid
        # storing the entire CC trace in memory).  Fall back to '_cc_trace' if
        # a caller still passes it, but this path is no longer used by default.
        if qc_cfg.get('compute_cc_sharpness', True):
            precomputed = detection.get('_cc_sharpness')
            if precomputed is not None:
                qc['cc_sharpness'] = precomputed
            else:
                cc_trace = detection.get('_cc_trace')
                peak_idx = detection.get('time_index', 0)
                half_w = int(qc_cfg.get('cc_sharpness_window', 20)) // 2
                if cc_trace is not None and peak_idx is not None:
                    lo = max(0, peak_idx - half_w)
                    hi = min(len(cc_trace), peak_idx + half_w + 1)
                    window = cc_trace[lo:hi]
                    if len(window) > 3:
                        qc['cc_sharpness'] = float(window.max() - np.mean(window))
                    else:
                        qc['cc_sharpness'] = np.nan
                else:
                    qc['cc_sharpness'] = np.nan
        else:
            qc['cc_sharpness'] = np.nan

        return qc

    def _choose_channel_trace(self, stream: Stream, channel: str):
        """Return a Trace from `stream` matching `channel` (e.g. 'E','Z','N').
        Falls back to the first trace if requested channel not present.
        Returns None if stream is empty.
        """
        if stream is None or len(stream) == 0:
            return None
        # Try exact channel match (wildcard for network/station/location)
        try:
            sel = stream.select(channel=f"*{channel}")
            if len(sel) > 0:
                return sel[0]
        except Exception:
            pass
        # Fall back to first available trace
        return stream[0]
    
    def get_one_day_waveforms(self, time: UTCDateTime, 
                             station: str, 
                             channel: str, 
                             template_sr: float) -> Optional[Stream]:
        """Load and preprocess one day of continuous data.

        Loads extra padding before/after the target day so that the
        bandpass filter edge effects (ringing) do not contaminate the
        actual detection window.  After filtering the padded data is
        trimmed back to the requested day.
        """
        padding = float(self.config.get('detection_qc.edge_margin_seconds', 10.0))
        # Always use at least 120 s padding to absorb filter transient
        padding = max(padding, 120.0)

        st = self.load_waveforms(
            time - padding, time + 86400 + padding, station, channel
        )
        
        if st is None or len(st) == 0:
            return None
        
        if sum(tr.stats.npts for tr in st) < self.MIN_TOTAL_NPTS:
            return None

        # Merge traces to avoid gaps at day boundaries, fill tiny gaps
        st.merge(method=1, fill_value=0, interpolation_samples=-1)

        # Detrend + taper to suppress filter ringing at trace edges
        st.detrend('demean')
        st.detrend('linear')
        st.taper(max_percentage=None, max_length=padding * 0.5)
        
        # Apply filter on the padded data
        st = self.apply_bandpass_filter(
            st,
            self.config['template_creation.filter_min'],
            self.config['template_creation.filter_max']
        )

        # Trim back to the actual day (remove filter transients)
        st.trim(time, time + 86400, nearest_sample=True)
        
        # Resample if needed
        if st[0].stats.sampling_rate != template_sr:
            st.resample(template_sr)
        
        return st
    
    def detector_filter(self, detections: List[Dict], 
                       trace, 
                       min_spike_ratio: float) -> List[Dict]:
        """Filter out spike-like artifacts using a two-criteria approach.
        
        1. Peak-to-noise ratio: peak amplitude (±0.5 s) vs. MAD-based
           noise estimate from a longer reference window (±10 s).
        2. Impulsiveness: fraction of short-window samples exceeding
           30 % of the peak amplitude.  Real seismic events spread
           energy over many samples (fraction > 0.10), while spikes
           concentrate it in 1–3 samples (fraction < 0.05).
        
        A detection is only rejected as a spike if BOTH its peak/noise
        ratio exceeds ``min_spike_ratio`` AND the signal is impulsive.
        This prevents strong but real events from being discarded.
        """
        filtered = []
        n_no_data = 0
        n_spike = 0
        ratios = []
        
        # Robust noise scaling: MAD * 1.4826 ≈ std for Gaussian data
        MAD_SCALE = 1.4826
        noise_half_win = 10.0   # seconds for noise estimate
        impulsive_thresh = 0.10  # min fraction of samples above 30% peak
        
        for det in detections:
            t = det["time"]
            
            # Short window for peak measurement
            tr_short = trace.slice(t - 0.5, t + 0.5)
            if tr_short.stats.npts < 10:
                n_no_data += 1
                continue
            
            data_short = tr_short.data
            peak = float(np.max(np.abs(data_short)))
            
            # Longer window for robust noise estimate (MAD-based)
            tr_long = trace.slice(t - noise_half_win, t + noise_half_win)
            if tr_long.stats.npts < 50:
                # Fall back to short window std
                noise_est = float(np.std(data_short))
            else:
                mad = float(np.median(np.abs(tr_long.data - np.median(tr_long.data))))
                noise_est = mad * MAD_SCALE
            
            if noise_est == 0:
                n_no_data += 1
                continue
            
            ratio = peak / noise_est
            ratios.append(ratio)
            
            if ratio >= min_spike_ratio:
                # High amplitude — check if it's impulsive (spike-like)
                n_above = int(np.sum(np.abs(data_short) > 0.3 * peak))
                frac_above = n_above / len(data_short)
                if frac_above < impulsive_thresh:
                    # Impulsive: very few samples carry the energy → spike
                    n_spike += 1
                    continue
            
            filtered.append(det)
        
        # Diagnostic logging
        n_total = len(detections)
        n_kept = len(filtered)
        if n_total > 0:
            ratio_str = ""
            if ratios:
                ratio_str = (f" (peak/noise ratios: "
                            f"min={min(ratios):.1f}, "
                            f"median={np.median(ratios):.1f}, "
                            f"max={max(ratios):.1f})")
        
        return filtered
    
    def measure_event_amplitude(self, z_trace, 
                               time: UTCDateTime, 
                               pre: float, 
                               post: float) -> float:
        """Measure event amplitude from Z component with robust method."""
        tr_cut = z_trace.slice(time - pre, time + post)
        if tr_cut.stats.npts < 10:
            return np.nan
        # Optional detrend/taper to stabilize amplitude
        try:
            tr_cut.detrend('linear')
            tr_cut.taper(max_percentage=0.05)
        except Exception:
            pass
        method = self.config.get('template_matching.amplitude_method', 'max')
        pct = float(self.config.get('template_matching.amplitude_percentile', 95.0))
        return compute_amplitude(tr_cut.data, tr_cut.stats.sampling_rate, method=method, percentile=pct)

    def measure_event_amplitudes(self, z_trace,
                                 time: UTCDateTime,
                                 pre: float,
                                 post: float) -> Dict[str, float]:
        """Measure multiple amplitude representations for the event window."""
        tr_cut = z_trace.slice(time - pre, time + post)
        if tr_cut.stats.npts < 10:
            return {
                'event_amplitude': np.nan,
                'event_amplitude_max': np.nan,
                'event_amplitude_rms': np.nan,
                'event_amplitude_percentile': np.nan,
                'event_amplitude_envelope': np.nan,
            }
        try:
            tr_cut.detrend('linear')
            tr_cut.taper(max_percentage=0.05)
        except Exception:
            pass
        pct = float(self.config.get('template_matching.amplitude_percentile', 95.0))
        sr = tr_cut.stats.sampling_rate
        a_max = compute_amplitude(tr_cut.data, sr, method='max', percentile=pct)
        a_rms = compute_amplitude(tr_cut.data, sr, method='rms', percentile=pct)
        a_pctl = compute_amplitude(tr_cut.data, sr, method='percentile', percentile=pct)
        a_env = compute_amplitude(tr_cut.data, sr, method='envelope', percentile=pct)
        # Default amplitude follows configured method
        method = self.config.get('template_matching.amplitude_method', 'max')
        a_def = compute_amplitude(tr_cut.data, sr, method=method, percentile=pct)
        return {
            'event_amplitude': a_def,
            'event_amplitude_max': a_max,
            'event_amplitude_rms': a_rms,
            'event_amplitude_percentile': a_pctl,
            'event_amplitude_envelope': a_env,
        }
    
    def template_to_vector(self, tr, nfft: int = 1024):
        """Convert template to spectral feature vector."""
        data = tr.data.astype(np.float32)
        data -= data.mean()
        std = np.std(data)
        if std > 0:
            data /= std
        
        if len(data) < nfft:
            data = np.pad(data, (0, nfft - len(data)))
        else:
            data = data[:nfft]
        
        spec = np.abs(np.fft.rfft(data))
        spec /= np.linalg.norm(spec) + 1e-12
        return spec
        
    def cluster_templates(self, template_list: List, eps: float = 0.15):
        """Cluster templates and select representatives.
        
        For each cluster:
        - If cluster has only 1 template (singleton): keep it as-is
        - If cluster has multiple templates: keep only the representative (medoid)
        """
        vectors = []
        primary = str(self.config.get('stations.primary_channel', 'Z'))
        for tpl in template_list:
            tr = self._choose_channel_trace(tpl[0], primary)
            if tr is None:
                continue
            vectors.append(self.template_to_vector(tr))
        
        X = np.vstack(vectors)
        dist = cosine_distances(X)
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=1,
            metric="precomputed"
        ).fit(dist)
        
        clusters = {}
        for idx, lbl in enumerate(clustering.labels_):
            clusters.setdefault(lbl, []).append(idx)
        
        repr_templates = []
        repr_map = {}
        
        for cid, indices in clusters.items():
            if len(indices) == 1:
                medoid = indices[0]
            else:
                sub = dist[np.ix_(indices, indices)]
                medoid = indices[np.argmin(sub.sum(axis=1))]
            
            rid = len(repr_templates)
            repr_templates.append(template_list[medoid])
            repr_map[rid] = indices
        
        return repr_templates, repr_map

    def _normalized_xcorr_fft(self, signal: np.ndarray,
                            template: np.ndarray) -> np.ndarray:
        """
        Normalized cross-correlation using FFT.
        Fixed version with improved numerical stability.
        """

        # Use float64 for all calculations to avoid precision issues
        signal = signal.astype(np.float64)
        template = template.astype(np.float64)

        # Center the signals
        template = template - template.mean()
        signal = signal - signal.mean()

        # Check for zero-variance template
        template_energy = np.sum(template ** 2)
        if template_energy < 1e-20:
            console.print(f"[red]WARNING: Template has near-zero energy: {template_energy:.2e}[/red]")
            return np.full(len(signal) - len(template) + 1, 0.0, dtype=np.float32)

        # FFT-based convolution
        corr = fftconvolve(signal, template[::-1], mode="valid")

        # Sliding window energy calculation (float64 precision)
        window_energy = np.convolve(signal ** 2, np.ones(len(template)), mode="valid")
        
        # Avoid division by zero in denominator
        window_energy = np.maximum(window_energy, 1e-20)
        
        # Normalization with full float64 precision
        denom = np.sqrt(template_energy * window_energy)
        result = corr / denom
        
        # Ensure result is bounded to [-1, 1] due to numerical precision
        result = np.clip(result, -1.0, 1.0)
        
        # Convert to float32 for memory efficiency
        return result.astype(np.float32)

    def _get_wavelet_widths(self, sr: float):
        cfg = self.config.get('template_matching.wavelet', {})
        num_scales = int(cfg.get('num_scales', 12))
        min_period = float(cfg.get('min_period', 0.05))
        max_period = float(cfg.get('max_period', 0.5))
        wavelet_w = float(cfg.get('wavelet_w', 6.0))
        periods = np.geomspace(min_period, max_period, num_scales)
        widths = (wavelet_w * sr * periods) / (2.0 * np.pi)
        return widths, wavelet_w

    def _wavelet_corr(self, signal: np.ndarray, template: np.ndarray, sr: float) -> np.ndarray:
        try:
            from scipy.signal import cwt as sp_cwt, morlet2 as sp_morlet2
        except Exception as e:
            raise RuntimeError(f"Wavelet mode requires scipy.signal.cwt/morlet2: {e}")

        signal = signal.astype(np.float64)
        template = template.astype(np.float64)
        widths, w = self._get_wavelet_widths(sr)

        Wx = sp_cwt(signal, sp_morlet2, widths, w=w)
        Wy = sp_cwt(template, sp_morlet2, widths, w=w)

        # Per-scale normalized correlation and aggregate
        agg = None
        scale_weighting = str(self.config.get('template_matching.wavelet.scale_weighting', 'uniform')).lower()
        for s in range(Wx.shape[0]):
            x_s = Wx[s]
            y_s = Wy[s]
            # complex conjugate correlation
            corr_s = fftconvolve(x_s, y_s[::-1].conj(), mode="valid")
            tpl_energy = np.sum(np.abs(y_s) ** 2)
            win_energy = np.convolve(np.abs(x_s) ** 2, np.ones(len(y_s)), mode="valid")
            denom = np.sqrt(tpl_energy * win_energy)
            denom[denom == 0] = np.inf
            cc_s = corr_s / denom
            if scale_weighting == 'energy':
                w_s = tpl_energy + 1e-12
                cc_s = cc_s * w_s
            agg = cc_s if agg is None else agg + cc_s

        if scale_weighting == 'energy':
            # Normalize by sum of weights approximately (constant across lags)
            total_w = np.sum([
                np.sum(np.abs(sp_cwt(template, sp_morlet2, [widths[i]], w=w)[0]) ** 2)
                for i in range(len(widths))
            ]) + 1e-12
            agg = agg / total_w
        else:
            agg = agg / Wx.shape[0]

        # Real part is the meaningful correlation measure
        return np.real(agg)

    def _wavelet_corr_precomputed(self, Wx: np.ndarray, signal_len: int, template: np.ndarray, sr: float, widths: np.ndarray, w: float) -> np.ndarray:
        """Wavelet correlation using precomputed signal CWT Wx for speed.
        Wx shape: (num_scales, signal_len)
        Returns valid-mode correlation array.
        """
        try:
            from scipy.signal import cwt as sp_cwt, morlet2 as sp_morlet2
        except Exception as e:
            raise RuntimeError(f"Wavelet mode requires scipy.signal.cwt/morlet2: {e}")

        template = template.astype(np.float64)
        Wy = sp_cwt(template, sp_morlet2, widths, w=w)

        agg = None
        scale_weighting = str(self.config.get('template_matching.wavelet.scale_weighting', 'uniform')).lower()
        for s in range(Wx.shape[0]):
            x_s = Wx[s]
            y_s = Wy[s]
            corr_s = fftconvolve(x_s, y_s[::-1].conj(), mode="valid")
            tpl_energy = np.sum(np.abs(y_s) ** 2)
            win_energy = np.convolve(np.abs(x_s) ** 2, np.ones(len(y_s)), mode="valid")
            denom = np.sqrt(tpl_energy * win_energy)
            denom[denom == 0] = np.inf
            cc_s = corr_s / denom
            if scale_weighting == 'energy':
                w_s = tpl_energy + 1e-12
                cc_s = cc_s * w_s
            agg = cc_s if agg is None else agg + cc_s

        if scale_weighting == 'energy':
            total_w = np.sum([
                np.sum(np.abs(sp_cwt(template, sp_morlet2, [widths[i]], w=w)[0]) ** 2)
                for i in range(len(widths))
            ]) + 1e-12
            agg = agg / total_w
        else:
            agg = agg / Wx.shape[0]

        return np.real(agg)

    def _next_pow2(self, n: int) -> int:
        return 1 << (int(n - 1).bit_length())

    def _get_scattering(self, L: int) -> Optional[object]:
        cfg = self.config.get('template_matching.wst', {})
        J = int(cfg.get('J', 6))
        Q = int(cfg.get('Q', 8))
        max_order = int(cfg.get('max_order', 2))
        backend = str(cfg.get('backend', 'numpy')).lower()
        device = str(cfg.get('device', 'cpu')).lower()
        key = (L, J, Q, max_order, backend, device)
        if key not in self._scattering_cache:
            # Only scatseisnet is supported now; make it a hard requirement.
            try:
                import numpy as _np
                import scatseisnet as ss
                # Force scatseisnet wavelet backend to numpy to avoid cupy/cuda issues
                try:
                    import scatseisnet.wavelet as _wave
                    _wave.xp = _np
                except Exception:
                    pass
                from scatseisnet.network import ScatteringNetwork
            except Exception as e:
                raise RuntimeError(f"scatseisnet is required for WST domain but import failed: {e}")

            # Build layer kwargs list; use max_order to define number of layers
            layers = [dict(octaves=J, resolution=1, quality=float(Q)) for _ in range(max_order if max_order > 0 else 1)]
            bins = min(128, max(32, L))
            try:
                scat = ScatteringNetwork(*layers, bins=bins, sampling_rate=1.0, verbose=False)
            except Exception as e:
                raise RuntimeError(f"Failed to construct scatseisnet ScatteringNetwork: {e}")

            self._scattering_cache[key] = scat
            # Evict oldest entries if cache grows too large
            while len(self._scattering_cache) > self._MAX_SCATTERING_CACHE:
                oldest_key = next(iter(self._scattering_cache))
                del self._scattering_cache[oldest_key]
            console.print(f"[dim]WST backend=scatseisnet ScatteringNetwork initialized (L={L}, J={J}, Q={Q}, layers={len(layers)})[/dim]")
        return self._scattering_cache[key]

    def _wst_feature(self, x: np.ndarray, L: int) -> np.ndarray:
        cfg = self.config.get('template_matching.wst', {})
        backend = str(cfg.get('backend', 'numpy')).lower()
        device = str(cfg.get('device', 'cpu')).lower()
        x = x.astype(np.float32)
        if len(x) < L:
            pad = np.zeros(L, dtype=np.float32)
            pad[:len(x)] = x
            x = pad
        elif len(x) > L:
            x = x[:L]
        scat = self._get_scattering(L)
        # Detect scatseisnet ScatteringNetwork-like objects by capability
        try:
            import scatseisnet as ss
            ScatNetwork = getattr(ss, 'ScatteringNetwork', None)
        except Exception:
            ScatNetwork = None

        def _is_scattering_network(obj):
            return hasattr(obj, 'transform') or hasattr(obj, 'transform_segment') or hasattr(obj, 'banks')

        # Torch backend handling
        if backend == 'torch':
            try:
                import torch
                if device == 'cuda' and not torch.cuda.is_available():
                    device = 'cpu'

                # If scat is a kymatio-like torch scattering module, use it directly
                if hasattr(scat, '__call__') and 'torch' in type(scat).__module__:
                    xt = torch.from_numpy(x).to(device)
                    S = scat(xt)
                    if hasattr(S, 'ndim') and S.ndim > 1:
                        feat = torch.mean(S, dim=-1)
                    else:
                        feat = S
                    feat = feat.flatten()
                    norm = torch.linalg.norm(feat) + 1e-12
                    feat = (feat / norm).detach().cpu().numpy().astype(np.float32)
                    return feat

                # Otherwise try to run scatseisnet ScatteringNetwork with torch tensor
                if _is_scattering_network(scat):
                    # ScatteringNetwork is a numpy-based object; use its transform_segment
                    try:
                        bins = getattr(scat, 'bins', x.shape[-1])
                        if len(x) >= bins:
                            start = (len(x) - bins) // 2
                            seg = x[start:start + bins]
                        else:
                            seg = np.zeros(bins, dtype=x.dtype)
                            seg[: len(x)] = x
                        S_list = scat.transform_segment(seg, reduce_type=np.mean)
                    except Exception:
                        S_list = scat.transform(x.reshape(1, -1), reduce_type=np.mean)
                    vals = [np.asarray(s).ravel() for s in S_list]
                    arr = np.concatenate(vals) if len(vals) > 0 else np.array([])
                    norm = np.linalg.norm(arr) + 1e-12
                    return (arr / norm).astype(np.float32)
            except Exception as e:
                console.print(f"[yellow]Torch WST failed ({e}); falling back to NumPy backend[/yellow]")

        # NumPy backend or fallback
        # If scat is a ScatteringNetwork from scatseisnet
        try:
            if _is_scattering_network(scat):
                # scatseisnet expects input shape (bins,) for transform_segment
                try:
                    bins = getattr(scat, 'bins', x.shape[-1])
                    if len(x) >= bins:
                        start = (len(x) - bins) // 2
                        seg = x[start:start + bins]
                    else:
                        seg = np.zeros(bins, dtype=x.dtype)
                        seg[: len(x)] = x
                    S_list = scat.transform_segment(seg, reduce_type=np.mean)
                except Exception:
                    S_list = scat.transform(x.reshape(1, -1), reduce_type=np.mean)
                vals = [np.asarray(s).ravel() for s in S_list]
                arr = np.concatenate(vals) if len(vals) > 0 else np.array([])
                norm = np.linalg.norm(arr) + 1e-12
                return (arr / norm).astype(np.float32)
        except Exception:
            # fall through to kymatio-like handling
            pass

        # Kymatio-like API: call scat with numpy array if callable
        if callable(scat):
            S = scat(x)
        else:
            raise RuntimeError('Scattering backend object is not callable and not a ScatteringNetwork')
        feat = np.mean(S, axis=-1) if getattr(S, 'ndim', 1) > 1 else S
        feat = feat.ravel()
        norm = np.linalg.norm(feat) + 1e-12
        return (feat / norm).astype(np.float32)

    def _wst_detector(self,
                      signal: np.ndarray,
                      template: np.ndarray,
                      threshold: float,
                      distance_samples: int,
                      sr: float,
                      template_id: int,
                      t0: UTCDateTime) -> List[Dict]:
        cfg = self.config.get('template_matching.wst', {})
        hop_seconds = float(cfg.get('hop_seconds', 0.1))
        metric = str(cfg.get('metric', 'cosine')).lower()

        L = self._next_pow2(len(template))
        tpl_feat = self._wst_feature(template, L)

        step = max(1, int(hop_seconds * sr))
        n = len(signal)
        if n < L:
            return []

        # Build similarity series at hop resolution
        sims = []
        indices = []
        for start in range(0, n - L + 1, step):
            win = signal[start:start + L]
            feat = self._wst_feature(win, L)
            if metric == 'l2':
                sim = -float(np.linalg.norm(feat - tpl_feat))
            else:  # cosine similarity
                sim = float(np.dot(feat, tpl_feat))
            sims.append(sim)
            indices.append(start + L // 2)

        sims = np.asarray(sims, dtype=np.float32)
        # Find peaks in similarity series
        # Convert sample-based minimum distance to index (hop) distance
        distance_idx = max(1, int(distance_samples / max(step, 1)))
        peaks, props = find_peaks(sims, height=threshold, distance=distance_idx)
        dets = []
        for k in peaks:
            center_sample = int(indices[k])
            dets.append({
                "time_index": center_sample,
                "time": t0 + float(center_sample) / float(sr),
                "similarity": float(sims[k]),
                "template_id": template_id
            })
        return dets
    def correlation_detector(
        self,
        signal: np.ndarray,
        templates: List[np.ndarray],
        threshold: float,
        distance: int,
        template_ids: List[int] = None,
        sampling_rate: float = None
    ):
        if template_ids is None:
            template_ids = list(range(len(templates)))

        detections = []

        domain = str(self.config.get('template_matching.domain', 'fft')).lower()

        for tpl, tid in zip(templates, template_ids):

            if len(tpl) >= len(signal):
                continue

            if domain == 'wavelet' and sampling_rate is not None:
                try:
                    cc = self._wavelet_corr(signal, tpl, sampling_rate)
                except Exception:
                    cc = self._normalized_xcorr_fft(signal, tpl)
            else:
                cc = self._normalized_xcorr_fft(signal, tpl)

            # Find local peaks above threshold and respecting minimum distance
            # Edge margin: skip peaks near start/end to avoid CC normalization artifacts
            _edge_sec = float(self.config.get('detection_qc.edge_margin_seconds', 10.0))
            _sr = sampling_rate if sampling_rate is not None else 100.0
            _edge_samples = int(_edge_sec * _sr)

            peaks, props = find_peaks(cc, height=threshold, distance=distance)
            if len(peaks) == 0:
                continue

            for p in peaks:
                if p < _edge_samples or p > len(cc) - _edge_samples:
                    continue
                detections.append({
                    "time_index": int(p),
                    "similarity": float(cc[p]),
                    "template_id": tid
                })

        return detections, None

    
    def process_one_day(self, 
                    date: pd.Timestamp, 
                    station: str, 
                    channel: str, 
                    template_list: List,
                    height: float, 
                    distance: int, 
                    output_dir: Path,
                    extra_stations: Optional[List[Dict]] = None) -> bool:
        """Process a single day with optional multi-station/multi-component stacking.

        Parameters
        ----------
        extra_stations : list of station dicts, optional
            Additional stations to use for stacking.  Each dict must have
            keys ``code`` and ``channels``.  The primary station is always
            ``station``/``channel``.
        """
        try:
            template_sr = template_list[0][0][0].stats.sampling_rate

            # --- Determine which (station, channel) pairs to correlate on ---
            use_all_components = bool(self.config.get('stations.use_all_components', False))
            stacking_method = str(self.config.get('stations.stacking_method', 'mean')).lower()
            comp_weights_cfg = self.config.get('stations.component_weights', {})

            # Primary station channels
            pairs = []  # list of (station_code, channel_code, weight)
            if use_all_components:
                all_channels = self.config.get('stations.channels', ['Z', 'N', 'E'])
                for ch in all_channels:
                    w = float(comp_weights_cfg.get(ch, 1.0))
                    pairs.append((station, ch, w))
            else:
                pairs.append((station, channel, 1.0))

            # Additional stations
            if extra_stations:
                for sdef in extra_stations:
                    s_code = sdef['code']
                    s_weight = float(sdef.get('weight', 1.0))
                    if use_all_components:
                        for ch in sdef.get('channels', ['Z', 'N', 'E']):
                            w = float(comp_weights_cfg.get(ch, 1.0)) * s_weight
                            pairs.append((s_code, ch, w))
                    else:
                        pairs.append((s_code, channel, s_weight))

            # --- Load continuous data for all pairs ---
            pair_traces = {}  # (station, channel) -> Trace
            for (sta, ch, _w) in pairs:
                st = self.get_one_day_waveforms(UTCDateTime(date), sta, ch, template_sr)
                if st is not None:
                    tr = self._choose_channel_trace(st, ch)
                    if tr is not None:
                        tr.data = tr.data.astype(np.float32)
                        pair_traces[(sta, ch)] = tr

            if not pair_traces:
                console.print(f"[dim]No waveforms available for {date.date()} any station/channel[/dim]")
                return False

            # For amplitude measurement, prefer the primary (station, channel) trace
            primary_trace = pair_traces.get((station, channel))

            detections = []
            domain = str(self.config.get('template_matching.domain', 'fft')).lower()
            sr = None  # will be set from first available trace

            # Precompute CWT once per (station, channel) for wavelet mode
            wavelet_precomp = {}
            decim = int(self.config.get('template_matching.wavelet.decimate_factor', 1) or 1)
            if domain == 'wavelet':
                for key, tr in pair_traces.items():
                    sr_tr = float(tr.stats.sampling_rate)
                    x = tr.data.astype(np.float64)
                    if decim > 1:
                        try:
                            from scipy.signal import decimate as sp_decimate
                            x = sp_decimate(x, decim, ftype='fir', zero_phase=True)
                        except Exception:
                            x = x[::decim]
                    sr_eff = sr_tr / decim
                    widths, w = self._get_wavelet_widths(sr_eff)
                    try:
                        from scipy.signal import cwt as sp_cwt, morlet2 as sp_morlet2
                    except Exception as e:
                        raise RuntimeError(f"Wavelet mode requires scipy.signal.cwt/morlet2: {e}")
                    Wx = sp_cwt(x, sp_morlet2, widths, w=w)
                    # Store CWT as complex64 to halve memory (complex128 is overkill for CC)
                    Wx = Wx.astype(np.complex64)
                    wavelet_precomp[key] = (Wx, x, sr_eff, widths, w)

            # Determine sr and t0 from first available trace
            first_key = next(iter(pair_traces))
            first_tr = pair_traces[first_key]
            sr = float(first_tr.stats.sampling_rate)
            t0 = first_tr.stats.starttime

            for tidx, tpl in enumerate(template_list):
                template_stream = tpl[0]

                # --- Compute CC for each (station, channel) pair ---
                cc_list = []
                cc_weights = []
                for (sta, ch, w) in pairs:
                    if (sta, ch) not in pair_traces:
                        continue
                    # Get matching template trace
                    template_trace = self._choose_channel_trace(template_stream, ch)
                    if template_trace is None:
                        continue
                    template_trace.data = template_trace.data.astype(np.float32)
                    cont_trace = pair_traces[(sta, ch)]

                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='Cannot cast ufunc.*', category=RuntimeWarning)
                        warnings.filterwarnings('ignore', message='divide by zero', category=RuntimeWarning)

                        try:
                            if domain == 'wst':
                                # WST doesn't produce a smooth CC trace; fall through to peak detection individually
                                dets = self._wst_detector(
                                    cont_trace.data, template_trace.data,
                                    threshold=height, distance_samples=distance,
                                    sr=float(cont_trace.stats.sampling_rate),
                                    template_id=tidx, t0=cont_trace.stats.starttime)
                                # For WST, append detections directly (no stacking)
                                detections.extend(dets)
                                continue
                            elif domain == 'wavelet' and (sta, ch) in wavelet_precomp:
                                Wx, x, sr_eff, widths, w_val = wavelet_precomp[(sta, ch)]
                                temp = template_trace.data.astype(np.float64)
                                if decim > 1:
                                    try:
                                        from scipy.signal import decimate as sp_decimate
                                        temp = sp_decimate(temp, decim, ftype='fir', zero_phase=True)
                                    except Exception:
                                        temp = temp[::decim]
                                cc = self._wavelet_corr_precomputed(Wx, len(x), temp, sr_eff, widths, w_val)
                            else:
                                # Use GPU backend if enabled
                                gpu_enabled = bool(self.config.get('gpu.enabled', False))
                                if gpu_enabled:
                                    from .gpu_correlator import batch_xcorr
                                    gpu_backend = str(self.config.get('gpu.backend', 'cupy'))
                                    gpu_device = str(self.config.get('gpu.device', 'cuda:0'))
                                    cc_results = batch_xcorr(
                                        cont_trace.data, [template_trace.data],
                                        backend=gpu_backend, device=gpu_device,
                                    )
                                    cc = cc_results[0] if cc_results else self._normalized_xcorr_fft(cont_trace.data, template_trace.data)
                                else:
                                    cc = self._normalized_xcorr_fft(cont_trace.data, template_trace.data)

                            cc_list.append(cc)
                            cc_weights.append(w)
                        except Exception as e:
                            console.print(f"[dim]Template {tidx} ({sta}.{ch}) failed: {str(e)[:100]}[/dim]")
                            continue

                if not cc_list:
                    continue

                # --- Stack CC traces ---
                if len(cc_list) == 1:
                    stacked_cc = cc_list[0]
                else:
                    stacked_cc = self._stack_cc_traces(cc_list, cc_weights, stacking_method)
                if stacked_cc is None:
                    continue

                # --- Peak detection on stacked CC ---
                if domain == 'wavelet':
                    distance_idx = max(1, int(distance / max(decim, 1)))
                    sr_for_time = sr / decim
                else:
                    distance_idx = distance
                    sr_for_time = sr

                # Precompute QC sharpness window size
                _qc_half_w = int(self.config.get('detection_qc.cc_sharpness_window', 20)) // 2

                # Edge margin: skip peaks near start/end of the processing window
                # to avoid spurious detections from CC normalization artifacts
                _edge_sec = float(self.config.get('detection_qc.edge_margin_seconds', 10.0))
                _edge_samples = int(_edge_sec * sr_for_time)

                peaks, props = find_peaks(stacked_cc, height=height, distance=distance_idx)
                for p in peaks:
                    # Reject peaks within edge margin of data boundaries
                    if p < _edge_samples or p > len(stacked_cc) - _edge_samples:
                        continue
                    # Compute CC sharpness inline to avoid storing the entire CC array
                    _lo = max(0, p - _qc_half_w)
                    _hi = min(len(stacked_cc), p + _qc_half_w + 1)
                    _win = stacked_cc[_lo:_hi]
                    _cc_sharpness = float(_win.max() - np.mean(_win)) if len(_win) > 3 else np.nan

                    det = {
                        'time_index': int(p),
                        'time': t0 + float(p) / float(sr_for_time),
                        'similarity': float(stacked_cc[p]),
                        'template_id': tidx,
                        'n_stations': len(cc_list),
                        '_cc_sharpness': _cc_sharpness,  # precomputed, no CC array stored
                    }
                    
                    # Check for impossible correlation values (should be rare now)
                    cc_val = float(stacked_cc[p])
                    if cc_val > 1.0 or cc_val < -1.0:
                        console.print(f"[yellow]Note: Clipping correlation {cc_val:.3f} to valid range for template {tidx}[/yellow]")
                        det['similarity'] = float(np.clip(cc_val, -1.0, 1.0))
                    
                    detections.append(det)

                # Free stacked CC array immediately
                del stacked_cc
            
            # --- Free large intermediate data after template loop ---
            del cc_list, cc_weights
            if wavelet_precomp:
                del wavelet_precomp
            # Keep only primary_trace for amplitude/QC; free all other traces
            keys_to_del = [k for k in pair_traces if k != (station, channel)]
            for k in keys_to_del:
                del pair_traces[k]
            gc.collect()

            if not detections:
                return False
            
            # Filter artifacts (use the primary component trace)
            if primary_trace is not None:
                detections = self.detector_filter(
                    detections,
                    primary_trace,
                    self.config['template_matching.min_spike_ratio']
                )

            # QC-based filtering: min detection SNR
            min_det_snr = float(self.config.get('detection_qc.min_detection_snr', 0.0))

            if not detections:
                console.print(f"[dim]All detections filtered out by artifact filter for {date.date()}[/dim]")
                return False
            
            # Save detections
            rows = []
            for d in detections:
                amp_trace = primary_trace if primary_trace is not None else first_tr
                amps = self.measure_event_amplitudes(
                    amp_trace,
                    d["time"],
                    self.config['template_matching.pre_amplitude'],
                    self.config['template_matching.post_amplitude']
                ) if amp_trace is not None else {
                    'event_amplitude': np.nan,
                    'event_amplitude_max': np.nan,
                    'event_amplitude_rms': np.nan,
                    'event_amplitude_percentile': np.nan,
                    'event_amplitude_envelope': np.nan,
                }

                # QC metrics (cc_sharpness is precomputed; only SNR needs trace)
                qc = self._compute_detection_qc(amp_trace, d, sr)

                # Use precomputed cc_sharpness (avoids storing full CC array)
                if '_cc_sharpness' in d:
                    qc['cc_sharpness'] = d['_cc_sharpness']

                # Filter by minimum detection SNR
                if min_det_snr > 0 and np.isfinite(qc.get('detection_snr', np.nan)):
                    if qc['detection_snr'] < min_det_snr:
                        continue

                row = {
                    "time": pd.to_datetime(d["time"].datetime),
                    "template_id": d["template_id"],
                    "similarity": d["similarity"],
                    "n_stations": d.get("n_stations", 1),
                }
                row.update(amps)
                row.update(qc)
                rows.append(row)
            
            if not rows:
                console.print(f"[dim]No detections survived QC filtering for {date.date()}[/dim]")
                return False

            df = pd.DataFrame(rows)
            out_file = output_dir / f"detections_{date.strftime('%Y%m%d')}.csv"
            df.to_csv(out_file, index=False, float_format="%.6e")

            # Final cleanup of this day's data
            del detections, rows, df
            del pair_traces, primary_trace
            gc.collect()
            
            return True
            
        except Exception as e:
            console.print(f"[yellow]Warning for {date}: {e}[/yellow]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False

    # ------------------------------------------------------------------
    # Template creation / re-creation from configured waveform backend
    # ------------------------------------------------------------------

    def create_templates(self,
                         templates_dir: Path,
                         info_csv: Optional[Path] = None,
                         overwrite: bool = False) -> Dict:
        """Create or re-create template waveforms from the configured data backend.

        Reads event times from *template_info.csv* (or *info_csv*), loads
        waveforms from the SDS archive (or Squirrel), applies detrend → taper
        → bandpass with generous padding, then trims to the template window
        and saves each template as a MiniSEED file.

        This ensures the template waveforms are extracted **from the same data
        source** used for continuous day processing, avoiding data-provenance
        mismatches that silently destroy CC values.

        Parameters
        ----------
        templates_dir : Path
            Output directory for template MiniSEED files.
        info_csv : Path, optional
            Path to template_info.csv.  Defaults to
            ``templates_dir / template_info.csv``.
        overwrite : bool
            If True, re-create all templates even if the file exists.

        Returns
        -------
        dict
            Summary with keys ``n_total``, ``n_created``, ``n_skipped``,
            ``n_failed``.
        """
        import re as _re
        import yaml as _yaml

        templates_dir = Path(templates_dir)
        templates_dir.mkdir(parents=True, exist_ok=True)

        if info_csv is None:
            info_csv = templates_dir / self.config.get(
                'paths.template_info_file', 'template_info.csv')

        if not info_csv.exists():
            console.print(f"[red]Error: template info file not found: {info_csv}[/red]")
            return {"n_total": 0, "n_created": 0, "n_skipped": 0, "n_failed": 0}

        info_df = pd.read_csv(info_csv)

        # Determine event times — either from 'event_time' column or the filename
        if 'event_time' not in info_df.columns:
            def _parse_event_time(fname):
                m = _re.search(r'(\d{8})_(\d{6})', str(fname))
                if not m:
                    return None
                d, t = m.group(1), m.group(2)
                return UTCDateTime(f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}")
            info_df['event_time'] = info_df['file'].apply(_parse_event_time)
        else:
            info_df['event_time'] = info_df['event_time'].apply(
                lambda x: UTCDateTime(x) if pd.notna(x) else None)

        station = self.config.get('stations.station_code', 'XXXX')
        channels = self.config.get('stations.channels', ['Z', 'N', 'E'])
        network = self.config.get('stations.network', '*')
        pre = float(self.config.get('template_creation.pre_event', 0.5))
        post = float(self.config.get('template_creation.post_event', 5.0))
        fmin = float(self.config.get('template_creation.filter_min', 1.0))
        fmax = float(self.config.get('template_creation.filter_max', 30.0))
        target_sr = float(self.config.get('stations.sampling_rate', 100.0))

        # Travel-time estimation for shifting template window to P-arrival
        from ..utils.traveltime import compute_p_traveltime as _compute_tt
        station_lat = float(self.config.get('stations.lat', 0.0))
        station_lon = float(self.config.get('stations.lon', 0.0))
        velocity_model = self.config.get('template_creation.velocity_model', None)
        use_traveltime = (station_lat != 0.0 and station_lon != 0.0)

        # Use the same padding as get_one_day_waveforms so that the taper +
        # filter produce identical waveforms in the template window.
        day_padding = max(
            float(self.config.get('detection_qc.edge_margin_seconds', 10.0)),
            120.0)
        padding = max(day_padding, 5.0 / fmin)  # at least 5 cycles of the lowest freq

        console.print(f"[cyan]📝 Creating templates from waveform backend "
                      f"({self.waveform_backend})[/cyan]")
        console.print(f"[dim]   station={station}, channels={channels}, "
                      f"pre={pre}s, post={post}s, filter={fmin}-{fmax} Hz, "
                      f"padding={padding:.0f}s[/dim]")
        if use_traveltime:
            console.print(
                f"[dim]   Travel-time correction enabled "
                f"(station {station_lat:.4f}°N {station_lon:.4f}°E, "
                f"model={'ak135' if not velocity_model else velocity_model})[/dim]"
            )

        n_total = len(info_df)
        n_created = 0
        n_skipped = 0
        n_failed = 0
        updated_rows = []

        for idx, row in info_df.iterrows():
            event_time = row.get('event_time')
            if event_time is None:
                n_failed += 1
                continue

            fname = row.get('file', '')
            out_path = templates_dir / fname
            if not overwrite and out_path.exists():
                n_skipped += 1
                updated_rows.append(row)
                continue

            try:
                # Compute P travel time to shift window from origin to P-arrival
                p_tt = 0.0
                if use_traveltime:
                    ev_lat = float(row.get('lat', 0.0))
                    ev_lon = float(row.get('lon', 0.0))
                    ev_depth = float(row.get('depth', 8000.0))
                    if np.isnan(ev_depth) or ev_depth <= 0:
                        ev_depth = 8000.0
                    if ev_lat != 0.0 and ev_lon != 0.0:
                        try:
                            p_tt = _compute_tt(
                                ev_lat, ev_lon, ev_depth,
                                station_lat, station_lon,
                                velocity_model,
                            )
                        except Exception:
                            p_tt = 0.0

                # P-arrival time = origin time + travel time
                p_arrival = event_time + p_tt

                # Load data with padding for filter settling
                tmin = p_arrival - pre - padding
                tmax = p_arrival + post + padding

                st_all = Stream()
                for ch in channels:
                    st_ch = self.load_waveforms(tmin, tmax, station, ch)
                    if st_ch is not None and len(st_ch) > 0:
                        st_all += st_ch

                if len(st_all) == 0:
                    console.print(f"[dim]  Template {idx}: no waveforms at "
                                  f"{event_time}[/dim]")
                    n_failed += 1
                    continue

                st_all.merge(method=1, fill_value=0, interpolation_samples=-1)

                # Pre-process: detrend → taper → bandpass (on padded window)
                st_all.detrend('demean')
                st_all.detrend('linear')
                st_all.taper(max_percentage=None, max_length=padding * 0.5)
                st_all = self.apply_bandpass_filter(st_all, fmin, fmax)

                # Trim to actual template window (centered on P-arrival)
                st_all.trim(p_arrival - pre, p_arrival + post,
                            nearest_sample=True)

                if len(st_all) == 0 or all(tr.stats.npts < 10 for tr in st_all):
                    n_failed += 1
                    continue

                # Resample if needed
                if abs(st_all[0].stats.sampling_rate - target_sr) > 0.01:
                    st_all.resample(target_sr)

                # Compute amplitude measures on Z-component (or first available)
                z_tr = self._choose_channel_trace(st_all, 'Z')
                z_data = z_tr.data if z_tr is not None else st_all[0].data
                amp_max = compute_amplitude(z_data, target_sr, 'max')
                amp_rms = compute_amplitude(z_data, target_sr, 'rms')
                amp_pct = compute_amplitude(z_data, target_sr, 'percentile', 95.0)
                amp_env = compute_amplitude(z_data, target_sr, 'envelope')

                # Build output filename if not set
                if not fname:
                    ts = event_time.strftime('%Y%m%d_%H%M%S')
                    ch_tag = ''.join(channels)
                    fname = f"template_{station}_{ts}_{ch_tag}.mseed"
                    out_path = templates_dir / fname

                st_all.write(str(out_path), format='MSEED')

                # Update row with new amplitude values
                new_row = row.copy()
                new_row['amplitude'] = amp_max
                new_row['amplitude_max'] = amp_max
                new_row['amplitude_rms'] = amp_rms
                new_row['amplitude_percentile'] = amp_pct
                new_row['amplitude_envelope'] = amp_env
                new_row['file'] = fname
                updated_rows.append(new_row)
                n_created += 1

            except Exception as exc:
                console.print(f"[dim]  Template {idx}: error — {exc}[/dim]")
                n_failed += 1

        # Save updated info CSV
        if updated_rows:
            out_df = pd.DataFrame(updated_rows)
            out_df.to_csv(info_csv, index=False)

        # Save processing metadata for reproducibility
        proc_meta = {
            'processing': {
                'filter_min': fmin,
                'filter_max': fmax,
                'pre_event': pre,
                'post_event': post,
                'padding': padding,
                'sampling_rate': target_sr,
                'backend': self.waveform_backend,
                'created_from_backend': True,
                'traveltime_correction': use_traveltime,
                'velocity_model': str(velocity_model) if velocity_model else 'ak135',
            }
        }
        meta_path = templates_dir / 'template_processing.yaml'
        try:
            with open(meta_path, 'w') as f:
                _yaml.dump(proc_meta, f, default_flow_style=False)
        except Exception:
            pass

        console.print(f"[green]✅ Templates: {n_created} created, {n_skipped} "
                      f"skipped, {n_failed} failed (of {n_total} total)[/green]")

        return {
            "n_total": n_total,
            "n_created": n_created,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
        }

    def match_templates(self, 
                        templates_dir: Path,
                        plot_tsne_pre: bool = False,
                        ignore_template_settings: bool = False,
                        bbox: Optional[tuple] = None,
                        region: Optional[str] = None) -> Dict:
        """Main template matching workflow."""
        console.print("[cyan]🔍 Template Matching[/cyan]")

        # Optionally load template processing metadata to ensure consistency
        meta_file = templates_dir / "template_processing.yaml"
        templates_prefiltered = False
        if not ignore_template_settings and meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = yaml.safe_load(f) or {}
                proc = meta.get("processing", {})
                templates_prefiltered = bool(proc.get("created_from_backend", False))
                for key_from, key_to, cast in [
                    ("filter_min", 'template_creation.filter_min', float),
                    ("filter_max", 'template_creation.filter_max', float),
                    ("pre_event", 'template_creation.pre_event', float),
                    ("post_event", 'template_creation.post_event', float),
                    ("amplitude_method", 'template_matching.amplitude_method', str),
                    ("amplitude_percentile", 'template_matching.amplitude_percentile', float),
                ]:
                    val = proc.get(key_from)
                    if val is not None:
                        self.config.update(key_to, cast(val))
                console.print("[dim]Loaded processing settings from template_processing.yaml for consistent matching[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: could not load template_processing.yaml: {e}[/yellow]")

        # Load templates
        template_files = sorted(templates_dir.glob("*.mseed"))
        # Optionally filter templates by bbox or region using the template info file
        info_file = templates_dir / self.config['paths.template_info_file']
        if (bbox is not None or (region is not None and region != "")) and info_file.exists():
            try:
                info_df = pd.read_csv(info_file)
                mask = pd.Series([True] * len(info_df))
                if bbox is not None:
                    try:
                        min_lat, max_lat, min_lon, max_lon = bbox
                        mask &= info_df['lat'].between(min_lat, max_lat)
                        mask &= info_df['lon'].between(min_lon, max_lon)
                    except Exception:
                        console.print('[yellow]Warning: invalid --bbox values; skipping bbox filtering[/yellow]')
                if region is not None and region != "":
                    if 'region' in info_df.columns:
                        mask &= info_df['region'].astype(str).str.contains(region, case=False, na=False)
                    else:
                        console.print('[yellow]Warning: template info has no `region` column; skipping region filtering[/yellow]')

                filtered_files = info_df.loc[mask, 'file'].tolist()
                if filtered_files:
                    template_files = [templates_dir / f for f in filtered_files if (templates_dir / f).exists()]
                    console.print(f"[dim]Filtered templates by bbox/region: {len(template_files)} files remain[/dim]")
                    # Save filtered template info for downstream steps (e.g., plotting)
                    try:
                        filtered_info = info_df.loc[mask].copy()
                        filtered_info_file = templates_dir / "filtered_template_info.csv"
                        filtered_info.to_csv(filtered_info_file, index=False)
                        console.print(f"[dim]Saved filtered template metadata: {filtered_info_file}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: could not save filtered template metadata: {e}[/yellow]")
                else:
                    console.print('[yellow]Warning: no templates match bbox/region filter; proceeding with all templates[/yellow]')
            except Exception as e:
                console.print(f'[yellow]Warning: failed to apply bbox/region filtering: {e} — proceeding with all templates[/yellow]')
        if not template_files:
            console.print("[red]No template files found![/red]")
            return {"success": False}

        console.print(f"📁 Loading [bold]{len(template_files)}[/bold] templates...")
        template_streams = []
        template_meta = []

        # Pre-load template info CSV once (instead of per-template)
        info_file = templates_dir / self.config['paths.template_info_file']
        info_df_cached = None
        if info_file.exists():
            try:
                info_df_cached = pd.read_csv(info_file)
            except Exception:
                info_df_cached = None

        for template_id, f in enumerate(template_files):
            st = read(str(f))
            if templates_prefiltered:
                # Templates were created with create_templates() from the same
                # backend — already properly detrended, tapered, and filtered
                # with generous padding.  Only demean to remove any residual DC.
                st.detrend('demean')
            else:
                # External templates: detrend + taper before bandpass to reduce
                # edge artifacts on short windows (5–6 s).
                st.detrend('demean')
                st.detrend('linear')
                st.taper(max_percentage=0.1)
                st = self.apply_bandpass_filter(
                    st,
                    self.config['template_creation.filter_min'],
                    self.config['template_creation.filter_max']
                )
            template_streams.append([st.copy()])

            # Extract metadata
            event_name = f.stem
            magnitude = np.nan
            lat = np.nan
            lon = np.nan

            if info_df_cached is not None:
                matching = info_df_cached[info_df_cached['file'] == f.name]
                if not matching.empty:
                    magnitude = matching.iloc[0].get('magnitude', np.nan)
                    lat = matching.iloc[0].get('lat', np.nan)
                    lon = matching.iloc[0].get('lon', np.nan)

            template_meta.append({
                "template_id": template_id,
                "event_name": event_name,
                "station": self.config['stations.station_code'],
                "component": self.config['stations.primary_channel'],
                "amplitude": float(np.max(np.abs(st[0].data))),
                "magnitude": magnitude,
                "lat": lat,
                "lon": lon,
                "file": f.name
            })

        # Optional: t-SNE preview BEFORE any SNR filtering
        try:
            if plot_tsne_pre:
                plots_dir = self.config.get_path('plots_dir')
                eps = self.config.get('template_matching.cluster_eps', 0.2)
                plot_templates_tsne(templates_dir, plots_dir, eps=eps, filename="templates_tsne_prefilter.png")
                console.print(f"🖼️ Saved pre-filter t-SNE: [cyan]{plots_dir / 'templates_tsne_prefilter.png'}[/cyan]")
        except Exception as e:
            console.print(f"[yellow]TSNE pre-filter visualization failed: {e}[/yellow]")

        # Attach SNR if available
        info_file = templates_dir / self.config['paths.template_info_file']
        if info_file.exists():
            try:
                info_df = pd.read_csv(info_file)
                snr_map = dict(zip(info_df['file'], info_df['snr'])) if 'snr' in info_df.columns else {}
                for row in template_meta:
                    row['snr'] = float(snr_map.get(row['file'], np.nan))
            except Exception:
                for row in template_meta:
                    row['snr'] = np.nan
        else:
            for row in template_meta:
                row['snr'] = np.nan

        # Filter by SNR
        min_snr = self.config.get('template_matching.min_snr', None)
        if min_snr is not None:
            before = len(template_streams)
            # Keep templates if their SNR is NaN (unknown) OR meets the min_snr threshold.
            # Previously templates with NaN SNR were always dropped; this caused unexpected
            # filtering when many templates lacked SNR values. Treat NaN as "unknown" and
            # preserve by default so a very small min_snr (e.g. -1000) won't remove them.
            keep_indices = [
                i for i, row in enumerate(template_meta)
                if (np.isnan(row.get('snr', np.nan)) or float(row.get('snr', np.nan)) >= float(min_snr))
            ]
            template_streams = [template_streams[i] for i in keep_indices]
            template_meta = [template_meta[i] for i in keep_indices]
            console.print(f"🔍 Filtered templates by SNR >= {min_snr}: {len(template_streams)}/{before} kept (NaN SNRs preserved)")

        # Cluster templates (can be disabled via config)
        cluster_enabled = bool(self.config.get('template_matching.cluster_enabled', True))
        if cluster_enabled:
            console.print("🔗 Clustering templates...")
            template_streams, repr_map = self.cluster_templates(
                template_streams,
                eps=self.config['template_matching.cluster_eps']
            )
            console.print(f"✅ Reduced to [bold]{len(template_streams)}[/bold] representative templates")
        else:
            console.print("[dim]Clustering disabled by config; keeping all templates[/dim]")
            # Identity mapping: each template is its own representative
            repr_map = {i: [i] for i in range(len(template_streams))}

        # Remap metadata: save ALL original template IDs mapped to their representative
        mapping_rows = []
        new_meta = []
        new_id = 0
        for _, orig_ids in repr_map.items():
            # Record mapping for ALL original IDs in this cluster
            for oid in orig_ids:
                mapping_rows.append({
                    "original_template_id": oid,
                    "representative_id": new_id,
                    "template_id": new_id,
                })
            # Keep representative metadata (first match as representative)
            for row in template_meta:
                if row["template_id"] in orig_ids:
                    r = row.copy()
                    r["original_template_id"] = row["template_id"]
                    r["representative_id"] = new_id
                    r["template_id"] = new_id
                    new_meta.append(r)
                    break
            new_id += 1

        mapping_df = pd.DataFrame(mapping_rows)
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        console.print(f"📋 Template ID mapping saved to: [cyan]{mapping_file}[/cyan] ({len(mapping_df)} entries)")

        # Output dir
        station = self.config['stations.station_code']
        channel = self.config['stations.primary_channel']
        output_dir = self.config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Dates
        start_date = pd.Timestamp(self.config['template_matching.start_date'])
        dates = pd.date_range(start=start_date, periods=self.config['template_matching.days_to_process'], freq="D")

        console.print(f"📅 Processing [bold]{len(dates)}[/bold] days...")
        n_jobs = self.config['template_matching.n_jobs']

        # Determine extra stations for multi-station stacking
        all_stations = self.get_station_list()
        extra_stations = [s for s in all_stations if s['code'] != station] if len(all_stations) > 1 else None
        if extra_stations:
            console.print(f"📡 Multi-station stacking with {len(all_stations)} stations: "
                          f"{[s['code'] for s in all_stations]}")

        # pre_dispatch limits queued jobs to reduce peak memory — each worker
        # holds one full day of waveform data plus CC arrays.
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as rich_progress:
            task_id = rich_progress.add_task(
                f"Matching {len(template_streams)} templates × {len(dates)} days",
                total=len(dates),
            )

            # Joblib callback to advance the Rich progress bar after each job
            import joblib
            class _RichProgressCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    rich_progress.advance(task_id, 1)
                    return super().__call__(*args, **kwargs)

            old_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = _RichProgressCallback
            try:
                results = Parallel(n_jobs=n_jobs, pre_dispatch='2*n_jobs')(
                    delayed(self.process_one_day)(
                        d, station, channel,
                        template_streams,
                        self.config['template_matching.similarity_threshold'],
                        self.config['template_matching.distance_samples'],
                        output_dir,
                        extra_stations=extra_stations,
                    )
                    for d in dates
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback

        successful_days = sum(results)
        console.print(f"✅ Template matching completed: [bold]{successful_days}/{len(dates)}[/bold] days successful")
        console.print(f"📁 Results saved to: [cyan]{output_dir}[/cyan]")

        return {
            "success": True,
            "days_processed": len(dates),
            "successful_days": successful_days,
            "output_dir": output_dir,
            "representative_templates": len(template_streams)
        }