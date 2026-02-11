"""
Template creation from catalog events.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from obspy import UTCDateTime, Stream
from rich.console import Console
from rich.progress import Progress

from .base import BaseProcessor
from ..config import Config
import yaml
from ..utils.helpers import compute_amplitude


console = Console()


class TemplateCreator(BaseProcessor):
    """Create templates from catalog events."""
    
    def _classic_sta_lta_np(self, data: np.ndarray, nsta: int, nlta: int) -> np.ndarray:
        """Lightweight STA/LTA using numpy moving averages on squared signal.
        Avoids importing obspy.signal to keep SciPy deps minimal.
        """
        if nsta <= 0 or nlta <= 0 or len(data) < max(nsta, nlta):
            return np.zeros_like(data, dtype=np.float32)
        x = np.asarray(data, dtype=np.float32)
        x2 = x * x
        # moving average via cumulative sum
        def mov_avg(arr, w):
            cs = np.cumsum(np.insert(arr, 0, 0.0))
            out = (cs[w:] - cs[:-w]) / float(w)
            # pad to original length (align center-left like obspy approx)
            pad_left = w // 2
            pad_right = len(arr) - len(out) - pad_left
            return np.pad(out, (pad_left, pad_right), mode='edge')
        sta = mov_avg(x2, nsta)
        lta = mov_avg(x2, nlta)
        cft = np.zeros_like(x, dtype=np.float32)
        nz = lta > 1e-12
        cft[nz] = (sta[nz] / lta[nz]).astype(np.float32)
        return cft

    def _trigger_onset_np(self, cft: np.ndarray, thr_on: float, thr_off: float):
        """Simple threshold trigger: returns list of [on, off] indices."""
        onsets = []
        active = False
        on_idx = 0
        for i, v in enumerate(cft):
            if not active and v >= thr_on:
                active = True
                on_idx = i
            elif active and v <= thr_off:
                active = False
                onsets.append([on_idx, i])
        if active:
            onsets.append([on_idx, len(cft) - 1])
        return onsets
    
    def detect_onset(self, trace) -> Optional[UTCDateTime]:
        """Detect P-wave onset in trace."""
        df = trace.stats.sampling_rate
        c_sta = int(self.config['template_creation.sta_window'] * df)
        c_lta = int(self.config['template_creation.lta_window'] * df)
        
        if len(trace.data) < c_lta:
            return None
        
        cft = self._classic_sta_lta_np(trace.data, c_sta, c_lta)
        on_off = self._trigger_onset_np(
            cft,
            float(self.config['template_creation.onset_thr_on']),
            float(self.config['template_creation.onset_thr_off'])
        )
        
        if len(on_off) == 0:
            return None
        
        return trace.stats.starttime + on_off[0][0] / df
    
    def extract_template(self, event_time: UTCDateTime, 
                        station: str, 
                        channel: str) -> Optional[Stream]:
        """Extract template waveform for an event."""
        # Load data around event
        tmin = event_time - 10
        tmax = event_time + 60
        
        st = self.load_waveforms(tmin, tmax, station, channel)
        if st is None or len(st) == 0:
            return None
        
        # Apply filter
        st = self.apply_bandpass_filter(
            st,
            self.config['template_creation.filter_min'],
            self.config['template_creation.filter_max']
        )
        
        # Get longest trace (merged)
        if len(st) > 1:
            st.merge(method=1, fill_value=0.0)
        
        tr_full = st[0] if len(st) > 0 else None
        if tr_full is None:
            return None
        
        # Detect onset
        onset = self.detect_onset(tr_full)
        if onset is None:
            return None

        # Determine noise window (fallback to pre_event if not set)
        noise_window = self.config.get('template_creation.noise_window', self.config['template_creation.pre_event'])

        # Compute noise RMS from pre-onset segment (ensure bounds)
        noise_start = max(tr_full.stats.starttime, onset - noise_window)
        noise_end = onset
        try:
            noise_tr = tr_full.slice(noise_start, noise_end - 0.1)
            noise_rms = np.sqrt(np.mean(noise_tr.data ** 2)) if len(noise_tr.data) > 0 else 0.0
        except Exception:
            noise_rms = 0.0

        # Extract template window
        tpl = tr_full.slice(
            onset - self.config['template_creation.pre_event'],
            onset + self.config['template_creation.post_event']
        )

        # Compute signal RMS and SNR, attach to stats
        try:
            signal_rms = np.sqrt(np.mean(tpl.data ** 2)) if len(tpl.data) > 0 else 0.0
            snr = float(signal_rms / noise_rms) if noise_rms > 0 else float('nan')
        except Exception:
            snr = float('nan')

        # attach SNR to the trace stats for downstream use
        try:
            tpl.stats.snr = snr
        except Exception:
            pass

        return tpl
    
    def create_templates(self, 
                        catalog_path: Path, 
                        catalog_type: str,
                        bbox: Tuple[float, float, float, float],
                        region: Optional[str] = None,
                        progress: Optional[Progress] = None) -> Dict:
        """Main template creation workflow."""
        
        console.print("[cyan]📋 Loading Catalog[/cyan]")
        catalog_df = self.load_catalog_events(catalog_path, catalog_type)
        catalog_df = self.filter_by_bbox(catalog_df, bbox)
        if region:
            catalog_df = self.filter_by_region(catalog_df, region)
        
        if len(catalog_df) == 0:
            console.print("[red]No events found in specified region![/red]")
            return {"success": False, "templates_created": 0}
        
        # Setup progress tracking
        task = progress.add_task("Creating templates...", total=len(catalog_df)) if progress else None
        
        # Create templates directory
        templates_dir = self.config.get_path('templates_dir')
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        station = self.config['stations.station_code']
        channels = self.config['stations.channels']
        template_rows = []
        
        for idx, row in catalog_df.iterrows():
            st_evt = Stream()
            
            for ch in channels:
                tr = self.extract_template(row["event_time"], station, ch)
                if tr is not None:
                    st_evt.append(tr)
            
            if len(st_evt) == 0:
                if progress:
                    progress.update(task, advance=1)
                continue
            
            # Save template
            tstr = row["event_time"].strftime("%Y%m%d_%H%M%S")
            event_name = row.get("event_name", f"{station}_{tstr}")
            fname = templates_dir / f"template_{station}_{tstr}_ZNE.mseed"
            st_evt.write(str(fname), format="MSEED")
            
            # Compute amplitude using configured method and SNR (if available)
            z_tr = st_evt.select(component="Z")
            if len(z_tr) > 0:
                # Optional detrend/taper to stabilize amplitude
                try:
                    z_tr[0].detrend('linear')
                    z_tr[0].taper(max_percentage=0.05)
                except Exception:
                    pass
            amp_method = self.config.get('template_creation.amplitude_method', 'max')
            amp_pct = float(self.config.get('template_creation.amplitude_percentile', 95.0))
            data_arr = z_tr[0].data if len(z_tr)>0 else np.array([])
            sr_val = z_tr[0].stats.sampling_rate if len(z_tr)>0 else 0.0
            # Compute all amplitude representations
            amp_max = compute_amplitude(data_arr, sr_val, method='max', percentile=amp_pct)
            amp_rms = compute_amplitude(data_arr, sr_val, method='rms', percentile=amp_pct)
            amp_pctl = compute_amplitude(data_arr, sr_val, method='percentile', percentile=amp_pct)
            amp_env = compute_amplitude(data_arr, sr_val, method='envelope', percentile=amp_pct)
            # Default amplitude column follows configured method
            amp = compute_amplitude(data_arr, sr_val, method=amp_method, percentile=amp_pct)
            snr = getattr(z_tr[0].stats, 'snr', np.nan) if len(z_tr) > 0 else np.nan
            
            template_rows.append({
                "template_id": len(template_rows),
                "event_name": event_name,
                "station": station,
                "amplitude": amp,
                "amplitude_max": amp_max,
                "amplitude_rms": amp_rms,
                "amplitude_percentile": amp_pctl,
                "amplitude_envelope": amp_env,
                "snr": snr,
                "magnitude": row.get("magnitude", np.nan),
                "lat": row.get("lat", np.nan),
                "lon": row.get("lon", np.nan),
                "file": fname.name,
            })
            
            if progress:
                progress.update(task, advance=1, 
                              description=f"Created template {len(template_rows)}/{len(catalog_df)}")
        
        # Save template info
        if template_rows:
            df_tpl = pd.DataFrame(template_rows)
            info_file = templates_dir / self.config['paths.template_info_file']
            df_tpl.to_csv(info_file, index=False)

            # Persist processing settings used for creation to ensure consistency in match
            processing_meta = {
                "processing": {
                    "filter_min": float(self.config.get('template_creation.filter_min', 1.0)),
                    "filter_max": float(self.config.get('template_creation.filter_max', 30.0)),
                    "pre_event": float(self.config.get('template_creation.pre_event', 0.5)),
                    "post_event": float(self.config.get('template_creation.post_event', 5.0)),
                    "amplitude_method": str(self.config.get('template_creation.amplitude_method', 'max')),
                    "amplitude_percentile": float(self.config.get('template_creation.amplitude_percentile', 95.0)),
                    "station": str(self.config.get('stations.station_code', '')),
                    "channels": list(self.config.get('stations.channels', ['Z','N','E']))
                }
            }
            with open(templates_dir / "template_processing.yaml", "w") as f:
                yaml.safe_dump(processing_meta, f, default_flow_style=False, sort_keys=False)
            
            console.print(f"\n✅ Created [bold]{len(template_rows)}[/bold] templates")
            console.print(f"📁 Template info saved to: [cyan]{info_file}[/cyan]")
            
            return {
                "success": True,
                "templates_created": len(template_rows),
                "info_file": info_file,
                "templates_dir": templates_dir
            }
        
        console.print("[yellow]No templates were created![/yellow]")
        return {"success": False, "templates_created": 0}