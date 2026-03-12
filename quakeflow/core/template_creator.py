"""
Template creation from catalog events.
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from obspy import UTCDateTime, Stream, Trace
from rich.console import Console
from rich.progress import Progress

from .base import BaseProcessor
from ..config import Config
import yaml
from ..utils.helpers import compute_amplitude
from ..utils.traveltime import compute_p_traveltime


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

    def parse_snuffler_markers(self, path: Path) -> Dict[str, List[Tuple[UTCDateTime, UTCDateTime]]]:
        """Parse a pyrocko snuffler markers file and return paired start/end times per station.

        Expects lines like: `2019-08-07 21:13:45.93255  0 LE.OCHT..EHZ`.
        We ignore channel/location information and use the station token (second dot field).
        Pairs are created by sequential markers (0/1 -> start/end, 2/3 -> start/end ...).
        """
        markers = {}
        try:
            with open(path, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    time_str = parts[0] + ' ' + parts[1]
                    # try to find the token that looks like NET.STA.LOC.CHA (contains a dot)
                    token = None
                    for p in parts[2:]:
                        if '.' in p and len(p.split('.')) > 1:
                            token = p
                            break
                    # fallback: if nothing matching, try the last token
                    if token is None and len(parts) >= 3:
                        token = parts[-1]
                    try:
                        t = UTCDateTime(time_str)
                    except Exception:
                        # try parse ISO-like single token
                        try:
                            t = UTCDateTime(parts[0])
                        except Exception:
                            continue
                    # extract station name
                    station = token.split('.')[1] if '.' in token and len(token.split('.')) > 1 else token
                    markers.setdefault(station, []).append(t)
        except Exception:
            return {}

        # pair sequential markers into (start, end)
        paired = {}
        for sta, times in markers.items():
            times_sorted = sorted(times)
            pairs = []
            for i in range(0, len(times_sorted) - 1, 2):
                start = times_sorted[i]
                end = times_sorted[i+1]
                pairs.append((start, end))
            paired[sta] = pairs
        return paired

    def find_marker_pair(self, pairs: Dict[str, List[Tuple[UTCDateTime, UTCDateTime]]], station: str, evt_time: UTCDateTime, tol: Optional[float] = None):
        """Find the marker pair for a given event_time and station.

        Preference: a pair where start <= evt_time <= end. Otherwise nearest start within tol (seconds).
        """
        if not pairs or station not in pairs:
            return None
        cand = pairs[station]
        # first look for enclosing pair
        for (s, e) in cand:
            if s <= evt_time <= e:
                return (s, e)
        # otherwise find nearest start
        best = None
        best_dt = None
        for (s, e) in cand:
            dt = abs((s - evt_time))
            # UTCDateTime subtraction yields float seconds
            if best is None or dt < best_dt:
                best = (s, e)
                best_dt = dt
        if best is None:
            return None
        if tol is not None and best_dt > float(tol):
            return None
        return best

    # ------------------------------------------------------------------
    # Qseek marker / pick helpers
    # ------------------------------------------------------------------
    def parse_qseek_marker_file(
        self,
        marker_path: Path,
        station: str,
    ) -> Optional[UTCDateTime]:
        """Extract the P-wave pick time for *station* from a qseek per-event marker file.

        Qseek marker files (Snuffler v0.2 format) contain lines like::

            phase: 2026-01-01 01:01:53.967  0 2D.ALFL..*  <hash> 2026-01-01 01:01:49.360 Pmod  None None
            phase: 2026-01-01 01:01:52.128  1 BQ.BGG..*   <hash> 2026-01-01 01:01:49.360 Pobs  None  True

        We prefer ``Pobs`` (observed pick) over ``Pmod`` (model-predicted) and
        fall back to ``Sobs``/``Smod`` (S-wave) if no P is available.

        Returns the pick time as UTCDateTime, or None if nothing matches.
        """
        marker_path = Path(marker_path)
        if not marker_path.exists():
            return None

        p_obs = None
        p_mod = None
        s_obs = None
        s_mod = None

        try:
            with open(marker_path, 'r') as fh:
                for line in fh:
                    line = line.strip()
                    if not line.startswith('phase:'):
                        continue
                    parts = line.split()
                    # parts layout: phase: DATE TIME POLARITY NET.STA.LOC.CHA HASH EVENT_DATE EVENT_TIME PHASE_NAME ...
                    if len(parts) < 9:
                        continue

                    pick_time_str = parts[1] + ' ' + parts[2]
                    nslc_token = parts[4]  # e.g. 2D.ALFL..* or BQ.BGG..*
                    phase_name = parts[8]  # Pmod, Pobs, Smod, Sobs

                    # Check if this line belongs to our station
                    token_parts = nslc_token.split('.')
                    sta = token_parts[1] if len(token_parts) > 1 else nslc_token
                    if sta != station:
                        continue

                    try:
                        pick_t = UTCDateTime(pick_time_str)
                    except Exception:
                        continue

                    if phase_name == 'Pobs':
                        p_obs = pick_t
                    elif phase_name == 'Pmod':
                        p_mod = pick_t
                    elif phase_name == 'Sobs':
                        s_obs = pick_t
                    elif phase_name == 'Smod':
                        s_mod = pick_t
        except Exception:
            return None

        # Prefer observed P > modelled P > observed S > modelled S
        return p_obs or p_mod or s_obs or s_mod
    
    def extract_template(self, event_time: UTCDateTime, 
                        station: str, 
                        channel: str,
                        onset_override: Optional[UTCDateTime] = None,
                        end_override: Optional[UTCDateTime] = None,
                        preloaded_stream: Optional[Stream] = None) -> Optional[Stream]:
        """Extract template waveform for an event.
        
        Parameters
        ----------
        preloaded_stream : Stream, optional
            If provided, skip Squirrel query and use this stream (already
            filtered).  Avoids redundant I/O when creating multi-channel
            templates.
        """
        pre_ev = float(self.config.get('template_creation.pre_event', 0.5))
        post_ev = float(self.config.get('template_creation.post_event', 5.0))

        if preloaded_stream is not None:
            # Use the pre-loaded, already-filtered stream
            st = preloaded_stream.select(channel=f"*{channel}")
            if st is None or len(st) == 0:
                return None
            st = st.copy()  # don't mutate the caller's stream
        else:
            # Load only as much data as needed:
            #   STA/LTA needs lta_window before the onset, and we need
            #   post_event after.  A modest margin handles clock jitter.
            lta = float(self.config.get('template_creation.lta_window', 5.0))
            margin = 2.0  # seconds of extra padding
            tmin = event_time - lta - margin
            tmax = event_time + post_ev + margin

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
        
        # Determine onset: use override (from markers) if provided, otherwise detect
        if onset_override is not None:
            onset = onset_override
        else:
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

        # Extract desired template window.
        # If both onset_override and end_override are provided (markers),
        # use the marker start->end exactly (no STA/LTA and no padding).
        pre_ev = float(self.config.get('template_creation.pre_event', 0.5))
        post_ev = float(self.config.get('template_creation.post_event', 5.0))
        if onset_override is not None and end_override is not None:
            # Cut exactly from marker start to marker end
            desired_start = onset_override
            desired_end = end_override
            tpl = tr_full.slice(desired_start, desired_end)
            do_padding = False
        else:
            desired_start = onset - pre_ev
            desired_end = onset + post_ev
            tpl = tr_full.slice(desired_start, desired_end)
            do_padding = True

        # If padding is enabled (no marker end provided), pad to expected length
        try:
            sr = tpl.stats.sampling_rate
        except Exception:
            sr = tr_full.stats.sampling_rate
        expected_npts = int(round((pre_ev + post_ev) * sr))
        actual_npts = len(tpl.data) if tpl is not None else 0
        if do_padding and actual_npts < expected_npts:
            need = expected_npts - actual_npts
            try:
                pad = np.zeros(need, dtype=tpl.data.dtype if actual_npts>0 else np.float32)
                newdata = np.concatenate([tpl.data, pad]) if actual_npts>0 else pad
                tpl.data = newdata
                tpl.stats.npts = len(newdata)
                tpl.stats.endtime = tpl.stats.starttime + (len(newdata)-1)/sr
            except Exception:
                pass

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
        
        station_list = self.get_station_list()
        num_stations = len(station_list)
        # Debug: show resolved station list to help diagnose single-station runs
        try:
            console.print(f"Resolved station list ({num_stations}): {station_list}")
            console.print(f"Config stations block: {self.config.get('stations')}" )
            console.print(f"Top-level networks key: {self.config.get('networks', None)}")
        except Exception:
            pass
        # Adjust progress total when multiple stations are requested
        if progress and task is not None and num_stations > 1:
            try:
                progress.update(task, total=len(catalog_df) * num_stations)
            except Exception:
                pass
        # Optional snuffler markers file (pyrocko markers) to override onset/end times
        markers_file = self.config.get('template_creation.snuffler_markers_file', None)
        marker_tol = float(self.config.get('template_creation.marker_match_tolerance', 120.0))
        markers = None
        if markers_file:
            markers_path = Path(markers_file)
            if markers_path.exists():
                # Support either a single markers file or a directory of per-event marker files
                if markers_path.is_dir():
                    markers = {}
                    # iterate files and aggregate pairs per station
                    for mf in sorted(markers_path.iterdir()):
                        if not mf.is_file():
                            continue
                        try:
                            parsed = self.parse_snuffler_markers(mf)
                        except Exception:
                            parsed = {}
                        for sta, pairs in parsed.items():
                            markers.setdefault(sta, []).extend(pairs)
                    console.print(f"Using snuffler markers directory: {markers_path} (stations: {list(markers.keys())[:20]})")
                else:
                    markers = self.parse_snuffler_markers(markers_path)
                    console.print(f"Using snuffler markers from: {markers_path} (stations: {list(markers.keys())})")
            else:
                console.print(f"[yellow]Snuffler markers file not found: {markers_path} - ignoring[/yellow]")
        template_rows = []
        _accessor = 'quakeflow_template_creation'
        _use_sds = (self.waveform_backend == 'sds')
        if not _use_sds:
            sq = self.get_squirrel()

        # Travel-time model (possibly overridden per-station below)
        velocity_model = self.config.get('template_creation.velocity_model', None)

        # Loop over all configured stations
        for sdef in station_list:
            station = sdef.get('code')
            channels = sdef.get('channels', ['Z', 'N', 'E'])
            station_lat = float(sdef.get('lat', self.config.get('stations.lat', 0.0)))
            station_lon = float(sdef.get('lon', self.config.get('stations.lon', 0.0)))
            use_traveltime = (station_lat != 0.0 and station_lon != 0.0)
            if use_traveltime:
                console.print(
                    f"[dim]Travel-time correction enabled "
                    f"(station {station} {station_lat:.4f}°N {station_lon:.4f}°E, "
                    f"model={'ak135' if not velocity_model else velocity_model})[/dim]"
                )

            for ev_i, (idx, row) in enumerate(catalog_df.iterrows()):
                st_evt = Stream()

                # If markers available, find the station-specific marker pair for this event
                pair = None
                qseek_onset = None  # P-pick from qseek per-event marker file
                if markers is not None:
                    evt_time = row.get('event_time')
                    try:
                        evt_ut = UTCDateTime(evt_time)
                    except Exception:
                        evt_ut = row.get('event_time')
                    pair = self.find_marker_pair(markers, station, evt_ut, tol=marker_tol)

                # Qseek catalog: extract P-pick from per-event marker file
                if str(catalog_type).lower() == 'qseek' and pair is None:
                    mfile = row.get('_marker_file')
                    if mfile and str(mfile) != 'nan':
                        qseek_onset = self.parse_qseek_marker_file(Path(mfile), station)

                # Compute P travel time for generic catalogs (origin time only)
                p_traveltime = 0.0
                if pair is None and qseek_onset is None and use_traveltime:
                    ev_lat = float(row.get('lat', 0.0))
                    ev_lon = float(row.get('lon', 0.0))
                    ev_depth = float(row.get('depth', 8000.0))
                    if np.isnan(ev_depth) or ev_depth <= 0:
                        ev_depth = 8000.0
                    try:
                        p_traveltime = compute_p_traveltime(
                            ev_lat, ev_lon, ev_depth,
                            station_lat, station_lon,
                            velocity_model,
                        )
                    except Exception as e:
                        console.print(f"[dim]  Travel time error for event {ev_i}: {e}[/dim]")
                        p_traveltime = 0.0

                # --- Pre-load ALL channels in a single query -------------------
                pre_ev = float(self.config.get('template_creation.pre_event', 0.5))
                post_ev = float(self.config.get('template_creation.post_event', 5.0))
                lta = float(self.config.get('template_creation.lta_window', 5.0))
                margin = 2.0
                if pair is not None:
                    # When markers define the window, use marker bounds + margin
                    load_tmin = pair[0] - margin
                    load_tmax = pair[1] + margin
                elif qseek_onset is not None:
                    # Use qseek P-pick as center
                    load_tmin = qseek_onset - pre_ev - margin
                    load_tmax = qseek_onset + post_ev + margin
                else:
                    # Generic catalog: shift window by P travel time
                    load_tmin = row["event_time"] + p_traveltime - lta - margin
                    load_tmax = row["event_time"] + p_traveltime + post_ev + margin

                # One combined wildcard query for all channels at once
                if _use_sds:
                    preloaded = self.get_sds_client().get_waveforms(
                        network="*", station=station, location="*", channel="*",
                        starttime=load_tmin, endtime=load_tmax, merge=1,
                    )
                    for tr in preloaded:
                        if tr.data.dtype != np.float32:
                            tr.data = tr.data.astype(np.float32)
                else:
                    try:
                        tmin_f = float(load_tmin.timestamp) if hasattr(load_tmin, 'timestamp') and not callable(load_tmin.timestamp) else float(load_tmin.timestamp())
                        tmax_f = float(load_tmax.timestamp) if hasattr(load_tmax, 'timestamp') and not callable(load_tmax.timestamp) else float(load_tmax.timestamp())
                    except Exception:
                        tmin_f = float(load_tmin)
                        tmax_f = float(load_tmax)

                    raw_traces = sq.get_waveforms(
                        codes=[f"*.{station}.*.*"],
                        tmin=tmin_f,
                        tmax=tmax_f,
                        accessor_id=_accessor,
                    )
                    preloaded = Stream()
                    for tr in raw_traces:
                        try:
                            from pyrocko import obspy_compat
                            tr_obs = obspy_compat.to_obspy_trace(tr)
                            if tr_obs.data.dtype != np.float32:
                                tr_obs.data = tr_obs.data.astype(np.float32)
                            preloaded.append(tr_obs)
                        except Exception:
                            continue

                    # Release Squirrel's cached waveform data from previous batch
                    del raw_traces
                    sq.advance_accessor(_accessor, 'waveform')

                if len(preloaded) > 0:
                    try:
                        for tr in preloaded:
                            if tr.data.dtype != np.float32:
                                tr.data = tr.data.astype(np.float32)
                        preloaded.merge(method=1, fill_value=0.0)
                    except Exception:
                        pass
                    # Apply bandpass once for all channels
                    preloaded = self.apply_bandpass_filter(
                        preloaded,
                        self.config['template_creation.filter_min'],
                        self.config['template_creation.filter_max'],
                    )

                for ch in channels:
                    # Prefer marker pair overrides when available. If catalog is a DLF/picks
                    # type, treat the catalog event time as the pick onset and skip STA/LTA.
                    # For qseek, use the extracted P-pick as onset.
                    # For generic catalogs, use origin_time + P travel time as onset.
                    if pair is not None:
                        onset_override = pair[0]
                        end_override = pair[1]
                    elif qseek_onset is not None:
                        onset_override = qseek_onset
                        end_override = None
                    elif str(catalog_type).lower() in ('dlf', 'dat', 'picks'):
                        try:
                            onset_override = UTCDateTime(row.get('event_time'))
                        except Exception:
                            onset_override = row.get('event_time')
                        end_override = None
                    elif p_traveltime > 0.0:
                        # Generic catalog with travel-time correction
                        onset_override = row["event_time"] + p_traveltime
                        end_override = None
                    else:
                        onset_override = None
                        end_override = None
                    tr = self.extract_template(
                        row["event_time"], station, ch,
                        onset_override=onset_override,
                        end_override=end_override,
                        preloaded_stream=preloaded if len(preloaded) > 0 else None,
                    )
                    if tr is not None:
                        st_evt.append(tr)

                if len(st_evt) == 0:
                    if progress:
                        progress.update(task, advance=1)
                    # Periodic garbage collection
                    if ev_i > 0 and ev_i % 25 == 0:
                        gc.collect()
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

                # Periodic garbage collection to keep memory in check
                if ev_i > 0 and ev_i % 25 == 0:
                    gc.collect()
        
        # Release all cached waveform data
        if _use_sds:
            self.get_sds_client().clear_cache()
        else:
            sq.clear_accessor(_accessor)
        gc.collect()

        # Save template info
        if template_rows:
            df_tpl = pd.DataFrame(template_rows)
            info_file = templates_dir / self.config['paths.template_info_file']
            df_tpl.to_csv(info_file, index=False)

            # Persist processing settings used for creation to ensure consistency in match
            # Record stations used for template creation
            stations_meta = []
            try:
                for s in station_list:
                    stations_meta.append({
                        'code': s.get('code'),
                        'network': s.get('network', '*'),
                        'channels': list(s.get('channels', ['Z', 'N', 'E'])),
                        'lat': float(s.get('lat', 0.0)),
                        'lon': float(s.get('lon', 0.0)),
                    })
            except Exception:
                stations_meta = []

            processing_meta = {
                "processing": {
                    "filter_min": float(self.config.get('template_creation.filter_min', 1.0)),
                    "filter_max": float(self.config.get('template_creation.filter_max', 30.0)),
                    "pre_event": float(self.config.get('template_creation.pre_event', 0.5)),
                    "post_event": float(self.config.get('template_creation.post_event', 5.0)),
                    "amplitude_method": str(self.config.get('template_creation.amplitude_method', 'max')),
                    "amplitude_percentile": float(self.config.get('template_creation.amplitude_percentile', 95.0)),
                    "stations": stations_meta,
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