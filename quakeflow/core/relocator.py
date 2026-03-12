"""
Relative relocation support: export differential times for HypoDD / GrowClust.

This module computes differential travel-time measurements (dt.cc) from
cross-correlation of detection waveforms and writes them in standard formats
that can be ingested by relocation programs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from obspy import UTCDateTime, read
from scipy.signal import fftconvolve
from rich.console import Console
from itertools import combinations

from .base import BaseProcessor
from ..config import Config

console = Console()


class Relocator(BaseProcessor):
    """Compute differential times and export for relocation."""

    def _xcorr_dt(
        self,
        trace1: np.ndarray,
        trace2: np.ndarray,
        sr: float,
        max_shift_sec: float = 0.5,
    ) -> Tuple[float, float]:
        """Compute differential time and CC between two waveform snippets.

        Parameters
        ----------
        trace1, trace2 : 1D arrays of equal length
        sr : sampling rate (Hz)
        max_shift_sec : maximum allowed shift in seconds

        Returns
        -------
        dt : differential time (seconds), positive if trace2 is later
        cc : correlation coefficient at best lag
        """
        t1 = trace1.astype(np.float64)
        t2 = trace2.astype(np.float64)
        t1 -= t1.mean()
        t2 -= t2.mean()
        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 == 0 or n2 == 0:
            return 0.0, 0.0

        corr = fftconvolve(t1, t2[::-1], mode="full")
        corr /= n1 * n2

        max_shift_samples = int(max_shift_sec * sr)
        center = len(t1) - 1
        lo = max(0, center - max_shift_samples)
        hi = min(len(corr), center + max_shift_samples + 1)
        corr_window = corr[lo:hi]
        if len(corr_window) == 0:
            return 0.0, 0.0

        peak_idx = np.argmax(corr_window) + lo
        cc = float(corr[peak_idx])
        dt = float(peak_idx - center) / sr
        return dt, cc

    def compute_dt_cc(
        self,
        detections_df: pd.DataFrame,
        templates_dir: Path,
        max_pairs: int = 50,
        min_cc: float = 0.5,
        pre_window: float = 0.5,
        post_window: float = 5.0,
    ) -> pd.DataFrame:
        """Compute differential travel times between all detection pairs.

        For efficiency, only pairs sharing the same template or neighboring
        templates are considered (up to max_pairs per event).

        Returns a DataFrame with columns:
        ev_id1, ev_id2, station, dt, cc, phase
        """
        station = self.config.get('stations.station_code', 'XXXX')
        channel = self.config.get('stations.primary_channel', 'Z')

        df = detections_df.copy()
        if 'event_id' not in df.columns:
            df['event_id'] = np.arange(1, len(df) + 1)

        dt_rows = []

        # Group detections by template for efficient pairing
        for tpl_id, group in df.groupby('template_id'):
            if len(group) < 2:
                continue

            events = group.sort_values('time').reset_index(drop=True)
            n_events = len(events)

            # Pair selection: consecutive and near-in-time
            pairs = []
            for i in range(n_events):
                for j in range(i + 1, min(i + max_pairs + 1, n_events)):
                    pairs.append((i, j))

            for (i, j) in pairs:
                t1 = UTCDateTime(pd.Timestamp(events.iloc[i]['time']).to_pydatetime())
                t2 = UTCDateTime(pd.Timestamp(events.iloc[j]['time']).to_pydatetime())

                try:
                    st1 = self.load_waveforms(t1 - pre_window, t1 + post_window, station, channel)
                    st2 = self.load_waveforms(t2 - pre_window, t2 + post_window, station, channel)
                    if st1 is None or st2 is None or len(st1) == 0 or len(st2) == 0:
                        continue

                    d1 = st1[0].data.astype(np.float64)
                    d2 = st2[0].data.astype(np.float64)
                    sr = st1[0].stats.sampling_rate

                    # Ensure equal length
                    min_len = min(len(d1), len(d2))
                    d1, d2 = d1[:min_len], d2[:min_len]

                    dt, cc = self._xcorr_dt(d1, d2, sr)
                    if cc >= min_cc:
                        dt_rows.append({
                            'ev_id1': int(events.iloc[i]['event_id']),
                            'ev_id2': int(events.iloc[j]['event_id']),
                            'station': station,
                            'dt': dt,
                            'cc': cc,
                            'phase': 'P',
                        })
                except Exception:
                    continue

        return pd.DataFrame(dt_rows)

    def write_hypodd_dt_cc(
        self,
        dt_df: pd.DataFrame,
        output_file: Path,
    ):
        """Write dt.cc file in HypoDD format.

        Format:
        # ev_id1  ev_id2
        station  dt  cc  phase
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            # Group by event pairs
            for (id1, id2), group in dt_df.groupby(['ev_id1', 'ev_id2']):
                f.write(f"# {int(id1):8d} {int(id2):8d}\n")
                for _, row in group.iterrows():
                    f.write(f"{row['station']:7s} {row['dt']:9.6f} {row['cc']:7.4f} {row['phase']}\n")

        console.print(f"📍 HypoDD dt.cc written: [cyan]{output_file}[/cyan] "
                       f"({len(dt_df)} measurements, "
                       f"{dt_df[['ev_id1','ev_id2']].drop_duplicates().shape[0]} pairs)")

    def write_growclust_dt(
        self,
        dt_df: pd.DataFrame,
        output_file: Path,
    ):
        """Write dt file in GrowClust format."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            for _, row in dt_df.iterrows():
                f.write(f"{int(row['ev_id1']):8d} {int(row['ev_id2']):8d} "
                        f"{row['station']:7s} {row['dt']:10.6f} {row['cc']:7.4f} "
                        f"{row['phase']}\n")

        console.print(f"📍 GrowClust dt written: [cyan]{output_file}[/cyan] "
                       f"({len(dt_df)} measurements)")

    def write_event_catalog(
        self,
        detections_df: pd.DataFrame,
        output_file: Path,
    ):
        """Write event.dat catalog for HypoDD/GrowClust.

        Format per line:
        DATE  TIME  LAT  LON  DEPTH  MAG  HERR  VERR  RMS  ID
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df = detections_df.copy()
        if 'event_id' not in df.columns:
            df['event_id'] = np.arange(1, len(df) + 1)

        with open(output_file, 'w') as f:
            for _, row in df.iterrows():
                t = pd.Timestamp(row['time'])
                lat = row.get('lat', row.get('tpl_lat', 0.0))
                lon = row.get('lon', row.get('tpl_lon', 0.0))
                depth = row.get('depth', 10.0)
                mag = row.get('est_magnitude', 0.0)
                if pd.isna(lat) or pd.isna(lon):
                    continue
                f.write(
                    f"{t.strftime('%Y%m%d')}  "
                    f"{t.strftime('%H%M%S%f')[:10]}  "
                    f"{lat:9.5f}  {lon:10.5f}  "
                    f"{depth:6.2f}  {mag:5.2f}  "
                    f"0.00  0.00  0.00  "
                    f"{int(row['event_id']):8d}\n"
                )

        console.print(f"📋 Event catalog written: [cyan]{output_file}[/cyan] ({len(df)} events)")

    def relocate(
        self,
        detections_df: pd.DataFrame,
        templates_dir: Path,
        output_dir: Path,
    ) -> Dict:
        """Main relocation workflow: compute dt.cc and write files."""
        cfg = self.config.get('relocation', {})
        if not cfg.get('enabled', False):
            console.print("[dim]Relocation disabled in config[/dim]")
            return {'success': False, 'reason': 'disabled'}

        output_dir.mkdir(parents=True, exist_ok=True)

        max_pairs = int(cfg.get('max_dt_cc_pairs', 50))
        min_cc = float(cfg.get('min_cc_for_dt', 0.5))
        fmt = str(cfg.get('output_format', 'hypodd')).lower()

        console.print("[cyan]📍 Computing differential times for relocation...[/cyan]")
        dt_df = self.compute_dt_cc(
            detections_df, templates_dir,
            max_pairs=max_pairs, min_cc=min_cc,
        )

        if len(dt_df) == 0:
            console.print("[yellow]No dt measurements computed[/yellow]")
            return {'success': False, 'reason': 'no_measurements'}

        # Write outputs
        if fmt == 'growclust':
            self.write_growclust_dt(dt_df, output_dir / "dt.cc")
        else:
            self.write_hypodd_dt_cc(dt_df, output_dir / "dt.cc")

        self.write_event_catalog(detections_df, output_dir / "event.dat")

        # Save raw dt table
        dt_df.to_csv(output_dir / "dt_measurements.csv", index=False)

        return {
            'success': True,
            'n_measurements': len(dt_df),
            'n_pairs': dt_df[['ev_id1', 'ev_id2']].drop_duplicates().shape[0],
            'output_dir': str(output_dir),
        }
