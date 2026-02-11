"""
Template matching on continuous data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from obspy import Stream, UTCDateTime, read
# from obspy.signal.cross_correlation import correlation_detector
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from rich.console import Console
from rich.progress import Progress
from scipy.signal import fftconvolve, find_peaks
# Scattering backend (formerly Kymatio) is imported lazily in _get_scattering to avoid hard dependency at import time

from .base import BaseProcessor
from ..config import Config
from ..utils.helpers import compute_amplitude
import yaml
from ..utils.plotting import plot_templates_tsne


console = Console()


class TemplateMatcher(BaseProcessor):
    """Perform template matching on continuous data."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.MIN_TOTAL_NPTS = 500
        self._scattering_cache = {}
    
    def get_one_day_waveforms(self, time: UTCDateTime, 
                             station: str, 
                             channel: str, 
                             template_sr: float) -> Optional[Stream]:
        """Load and preprocess one day of continuous data."""
        st = self.load_waveforms(time, time + 86400, station, channel)
        
        if st is None or len(st) == 0:
            return None
        
        if sum(tr.stats.npts for tr in st) < self.MIN_TOTAL_NPTS:
            return None
        
        # Apply filter
        st = self.apply_bandpass_filter(
            st,
            self.config['template_creation.filter_min'],
            self.config['template_creation.filter_max']
        )
        
        # Resample if needed
        if st[0].stats.sampling_rate != template_sr:
            st.resample(template_sr)
        
        return st
    
    def detector_filter(self, detections: List[Dict], 
                       trace, 
                       min_spike_ratio: float) -> List[Dict]:
        """Filter out spike-like artifacts."""
        filtered = []
        
        for det in detections:
            t = det["time"]
            tr_cut = trace.slice(t - 0.5, t + 0.5)
            
            if tr_cut.stats.npts < 10:
                continue
            
            data = tr_cut.data
            std = np.std(data)
            if std == 0:
                continue
            # If the max amplitude compared to the local std is too large
            # it's likely a spike/artifact and should be rejected. Keep
            # detections that have a moderate peak-to-noise ratio.
            if np.max(np.abs(data)) / std < min_spike_ratio:
                filtered.append(det)
        
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
        for tpl in template_list:
            tr = tpl[0][0]  # Z component
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
        Equivalent to ObsPy correlation_detector core.
        """

        signal = signal.astype(np.float64)
        template = template.astype(np.float64)

        template -= template.mean()
        signal -= signal.mean()

        # FFT-based convolution
        corr = fftconvolve(signal, template[::-1], mode="valid")

        # normalization
        template_energy = np.sum(template ** 2)

        window_energy = np.convolve(
            signal ** 2,
            np.ones(len(template)),
            mode="valid"
        )

        denom = np.sqrt(template_energy * window_energy)
        denom[denom == 0] = np.inf

        return corr / denom

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
            peaks, props = find_peaks(cc, height=threshold, distance=distance)
            if len(peaks) == 0:
                continue

            for p in peaks:
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
                    output_dir: Path) -> bool:
        """Process a single day with fixed template structure."""
        try:
            st = self.get_one_day_waveforms(
                UTCDateTime(date),
                station,
                channel,
                template_sr=template_list[0][0][0].stats.sampling_rate
            )
            if st is None:
                return False
            
            try:
                # Get Z component as a single Trace
                z_stream = st.select(channel="*Z")
                if len(z_stream) == 0:
                    return False
                z_trace = z_stream[0]
            except IndexError:
                return False

            # Ensure trace data is float32
            z_trace.data = z_trace.data.astype(np.float32)

            detections = []
            domain = str(self.config.get('template_matching.domain', 'fft')).lower()

            # Precompute CWT of signal once for wavelet mode, with optional decimation
            sr = float(z_trace.stats.sampling_rate)
            t0 = z_trace.stats.starttime
            decim = int(self.config.get('template_matching.wavelet.decimate_factor', 1) or 1)
            if domain == 'wavelet':
                x = z_trace.data.astype(np.float64)
                if decim > 1:
                    try:
                        from scipy.signal import decimate as sp_decimate
                        x = sp_decimate(x, decim, ftype='fir', zero_phase=True)
                    except Exception:
                        x = x[::decim]
                sr_eff = sr / decim
                widths, w = self._get_wavelet_widths(sr_eff)
                try:
                    from scipy.signal import cwt as sp_cwt, morlet2 as sp_morlet2
                except Exception as e:
                    raise RuntimeError(f"Wavelet mode requires scipy.signal.cwt/morlet2: {e}")
                Wx = sp_cwt(x, sp_morlet2, widths, w=w)

            for tidx, tpl in enumerate(template_list):
                # tpl is [Stream], so tpl[0] is the Stream
                template_stream = tpl[0]

                # Get Z component from template (take first Z trace)
                template_z_stream = template_stream.select(channel="*Z")
                if len(template_z_stream) == 0:
                    continue
                template_z = template_z_stream[0]
                template_z.data = template_z.data.astype(np.float32)

                # Suppress warnings during correlation
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Cannot cast ufunc.*', category=RuntimeWarning)
                    warnings.filterwarnings('ignore', message='divide by zero', category=RuntimeWarning)

                    try:
                        if domain == 'wst':
                            dets = self._wst_detector(
                                z_trace.data,
                                template_z.data,
                                threshold=height,
                                distance_samples=distance,
                                sr=sr,
                                template_id=tidx,
                                t0=t0
                            )
                        elif domain == 'wavelet':
                            # Use precomputed CWT and effective sampling rate
                            temp = template_z.data.astype(np.float64)
                            if decim > 1:
                                try:
                                    from scipy.signal import decimate as sp_decimate
                                    temp = sp_decimate(temp, decim, ftype='fir', zero_phase=True)
                                except Exception:
                                    temp = temp[::decim]
                            cc = self._wavelet_corr_precomputed(Wx, len(x), temp, sr_eff, widths, w)
                            # Peaks on decimated grid
                            distance_idx = max(1, int(distance / max(decim, 1)))
                            peaks, props = find_peaks(cc, height=height, distance=distance_idx)
                            dets = []
                            for p in peaks:
                                dets.append({
                                    'time_index': int(p),
                                    'time': t0 + float(p) / float(sr_eff),
                                    'similarity': float(cc[p]),
                                    'template_id': tidx
                                })
                        else:
                            dets, _ = self.correlation_detector(
                                z_trace.data,
                                [template_z.data],
                                height,
                                distance,
                                template_ids=[tidx],
                                sampling_rate=sr
                            )
                            for d in dets:
                                if 'time_index' in d:
                                    d['time'] = t0 + float(d['time_index']) / float(sr)
                                d['template_id'] = tidx
                        detections.extend(dets)
                    except Exception as e:
                        console.print(f"[dim]Template {tidx} failed: {str(e)[:100]}[/dim]")
                        continue
            
            if not detections:
                return False
            
            # Filter artifacts (use the first trace from z_stream)
            if len(z_stream) > 0:
                detections = self.detector_filter(
                    detections, 
                    z_stream[0],  # Use Trace for filtering
                    self.config['template_matching.min_spike_ratio']
                )
            
            if not detections:
                return False
            
            # Save detections
            rows = []
            for d in detections:
                amps = self.measure_event_amplitudes(
                    z_stream[0] if len(z_stream) > 0 else None,
                    d["time"],
                    self.config['template_matching.pre_amplitude'],
                    self.config['template_matching.post_amplitude']
                ) if len(z_stream) > 0 else {
                    'event_amplitude': np.nan,
                    'event_amplitude_max': np.nan,
                    'event_amplitude_rms': np.nan,
                    'event_amplitude_percentile': np.nan,
                    'event_amplitude_envelope': np.nan,
                }
                row = {
                    "time": pd.to_datetime(d["time"].datetime),
                    "template_id": d["template_id"],
                    "similarity": d["similarity"],
                }
                row.update(amps)
                rows.append(row)
            
            df = pd.DataFrame(rows)
            out_file = output_dir / f"detections_{date.strftime('%Y%m%d')}.csv"
            df.to_csv(out_file, index=False, float_format="%.6e")
            
            return True
            
        except Exception as e:
            console.print(f"[yellow]Warning for {date}: {e}[/yellow]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return False
        
    def match_templates(self, 
                        templates_dir: Path,
                        progress: Optional[Progress] = None,
                        plot_tsne_pre: bool = False,
                        ignore_template_settings: bool = False,
                        bbox: Optional[tuple] = None,
                        region: Optional[str] = None) -> Dict:
        """Main template matching workflow."""
        console.print("[cyan]🔍 Template Matching[/cyan]")

        # Optionally load template processing metadata to ensure consistency
        meta_file = templates_dir / "template_processing.yaml"
        if not ignore_template_settings and meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = yaml.safe_load(f) or {}
                proc = meta.get("processing", {})
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
        for template_id, f in enumerate(template_files):
            st = read(str(f))
            # Apply filtering
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

            info_file = templates_dir / self.config['paths.template_info_file']
            if info_file.exists():
                info_df = pd.read_csv(info_file)
                matching = info_df[info_df['file'] == f.name]
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
            keep_indices = [i for i, row in enumerate(template_meta)
                            if (not np.isnan(row.get('snr', np.nan)) and float(row.get('snr', np.nan)) >= float(min_snr))]
            template_streams = [template_streams[i] for i in keep_indices]
            template_meta = [template_meta[i] for i in keep_indices]
            console.print(f"🔍 Filtered templates by SNR >= {min_snr}: {len(template_streams)}/{before} kept")

        # Cluster templates
        console.print("🔗 Clustering templates...")
        template_streams, repr_map = self.cluster_templates(
            template_streams,
            eps=self.config['template_matching.cluster_eps']
        )
        console.print(f"✅ Reduced to [bold]{len(template_streams)}[/bold] representative templates")

        # Remap metadata, keep mapping file
        new_meta = []
        new_id = 0
        for _, orig_ids in repr_map.items():
            for row in template_meta:
                if row["template_id"] in orig_ids:
                    r = row.copy()
                    r["original_template_id"] = row["template_id"]
                    r["representative_id"] = new_id
                    r["template_id"] = new_id
                    new_meta.append(r)
                    break
            new_id += 1

        mapping_df = pd.DataFrame([
            {
                "original_template_id": row["original_template_id"],
                "representative_id": row.get("representative_id"),
                "template_id": row["template_id"]
            }
            for row in new_meta
        ])
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        mapping_df.to_csv(mapping_file, index=False)
        console.print(f"📋 Template ID mapping saved to: [cyan]{mapping_file}[/cyan]")

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

        from tqdm.auto import tqdm
        from tqdm_joblib import tqdm_joblib
        with tqdm_joblib(tqdm(desc="Matching templates", total=len(dates))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.process_one_day)(
                    d, station, channel,
                    template_streams,
                    self.config['template_matching.similarity_threshold'],
                    self.config['template_matching.distance_samples'],
                    output_dir
                )
                for d in dates
            )

        successful_days = sum(results)
        console.print(f"\n✅ Template matching completed: [bold]{successful_days}/{len(dates)}[/bold] days successful")
        console.print(f"📁 Results saved to: [cyan]{output_dir}[/cyan]")

        return {
            "success": True,
            "days_processed": len(dates),
            "successful_days": successful_days,
            "output_dir": output_dir,
            "representative_templates": len(template_streams)
        }