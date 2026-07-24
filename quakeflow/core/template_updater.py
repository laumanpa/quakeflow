"""
Template self-updating: add high-quality detections as new templates.

After each detection run, events with high CC similarity and high SNR
can be promoted to templates to improve detection sensitivity in
subsequent iterations.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from obspy import UTCDateTime, Stream, read
from rich.console import Console

from .base import BaseProcessor
from ..config import Config
from ..utils.helpers import compute_amplitude

console = Console()


class TemplateUpdater(BaseProcessor):
    """Promote high-quality detections to templates."""

    def select_candidates(
        self,
        detections_df: pd.DataFrame,
        min_similarity: float = 0.8,
        min_snr: float = 5.0,
        max_candidates: int = 100,
    ) -> pd.DataFrame:
        """Select detections that qualify as new templates.

        Criteria:
        - CC similarity >= min_similarity
        - Detection SNR >= min_snr (if available)
        - Not too close in time to existing templates
        - Distributed across different time periods (diversity)
        """
        df = detections_df.copy()

        # Basic similarity filter
        mask = df['similarity'] >= min_similarity
        if 'detection_snr' in df.columns:
            snr_valid = df['detection_snr'].notna() & (df['detection_snr'] >= min_snr)
            mask = mask & (snr_valid | df['detection_snr'].isna())

        candidates = df[mask].copy()

        if len(candidates) == 0:
            return candidates

        # Sort by similarity (best first) and take top N
        candidates = candidates.sort_values('similarity', ascending=False)

        # Remove duplicates close in time (keep only the best per time cluster).
        # Greedy in similarity order: keep a candidate only if no already-kept
        # candidate lies within min_gap_sec.  Uses a sorted list of kept
        # timestamps + bisect for O(n log n) instead of the previous O(n^2).
        import bisect
        candidates = candidates.reset_index(drop=True)
        min_gap_sec = 60.0  # minimum 60 seconds between template events
        cand_ts = pd.to_datetime(candidates['time']).astype('int64').to_numpy() / 1e9  # epoch seconds
        kept_sorted = []  # sorted kept timestamps (seconds)
        keep = [False] * len(candidates)
        for i in range(len(candidates)):
            t_i = float(cand_ts[i])
            pos = bisect.bisect_left(kept_sorted, t_i)
            near = False
            if pos < len(kept_sorted) and (kept_sorted[pos] - t_i) < min_gap_sec:
                near = True
            if not near and pos > 0 and (t_i - kept_sorted[pos - 1]) < min_gap_sec:
                near = True
            if not near:
                keep[i] = True
                kept_sorted.insert(pos, t_i)
        candidates = candidates[keep]

        # Limit number
        candidates = candidates.head(max_candidates)

        console.print(f"🎯 Selected {len(candidates)} candidate detections for template promotion")
        return candidates

    def extract_and_save_templates(
        self,
        candidates_df: pd.DataFrame,
        templates_dir: Path,
        template_info_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """Extract waveforms for candidate detections and save as templates.

        Parameters
        ----------
        candidates_df : DataFrame with candidate detections
        templates_dir : directory to save new templates
        template_info_df : existing template info (to assign new IDs)

        Returns
        -------
        dict with 'n_added', 'new_info_df'
        """
        station = self.config.get('stations.station_code', 'XXXX')
        channels = self.config.get('stations.channels', ['Z', 'N', 'E'])
        pre_event = float(self.config.get('template_creation.pre_event', 0.5))
        post_event = float(self.config.get('template_creation.post_event', 5.0))
        filter_min = float(self.config.get('template_creation.filter_min', 1.0))
        filter_max = float(self.config.get('template_creation.filter_max', 30.0))

        # Determine next template ID
        next_id = 0
        if template_info_df is not None and len(template_info_df) > 0:
            next_id = int(template_info_df['template_id'].max()) + 1

        templates_dir.mkdir(parents=True, exist_ok=True)
        new_rows = []

        for _, row in candidates_df.iterrows():
            try:
                t = UTCDateTime(pd.Timestamp(row['time']).to_pydatetime())
                st = Stream()
                for ch in channels:
                    st_ch = self.load_waveforms(t - pre_event, t + post_event, station, ch)
                    if st_ch is not None:
                        st += st_ch

                if len(st) == 0:
                    continue

                # Apply filter
                st = self.apply_bandpass_filter(st, filter_min, filter_max)

                # Save
                tstr = t.strftime("%Y%m%d_%H%M%S")
                fname = templates_dir / f"template_{station}_{tstr}_ZNE_auto.mseed"
                st.write(str(fname), format="MSEED")

                # Compute amplitude
                z_tr = st.select(component="Z")
                amp = np.nan
                if len(z_tr) > 0:
                    amp = compute_amplitude(z_tr[0].data, z_tr[0].stats.sampling_rate, method='max')

                new_rows.append({
                    'template_id': next_id,
                    'event_name': f"{station}_{tstr}",
                    'station': station,
                    'amplitude': amp,
                    'magnitude': row.get('est_magnitude', np.nan),
                    'lat': row.get('lat', row.get('tpl_lat', np.nan)),
                    'lon': row.get('lon', row.get('tpl_lon', np.nan)),
                    'snr': row.get('detection_snr', np.nan),
                    'file': fname.name,
                    'source': 'auto_detected',
                    'parent_similarity': row.get('similarity', np.nan),
                })
                next_id += 1

            except Exception as e:
                console.print(f"[dim]Failed to extract template at {row['time']}: {e}[/dim]")
                continue

        new_info = pd.DataFrame(new_rows)
        console.print(f"✅ Added {len(new_info)} new auto-detected templates to {templates_dir}")

        return {
            'n_added': len(new_info),
            'new_info_df': new_info,
        }

    def update_template_catalog(
        self,
        templates_dir: Path,
        new_info_df: pd.DataFrame,
    ):
        """Append new template info to existing template_info.csv."""
        info_file = templates_dir / self.config.get('paths.template_info_file', 'template_info.csv')

        if info_file.exists():
            existing = pd.read_csv(info_file)
            # Avoid duplicates by file name
            existing_files = set(existing['file'].tolist())
            new_unique = new_info_df[~new_info_df['file'].isin(existing_files)]
            combined = pd.concat([existing, new_unique], ignore_index=True)
        else:
            combined = new_info_df

        combined.to_csv(info_file, index=False)
        console.print(f"📋 Updated template catalog: {info_file} ({len(combined)} total templates)")

    def update_templates(
        self,
        detections_df: pd.DataFrame,
        templates_dir: Path,
    ) -> Dict:
        """Main template updating workflow.

        Reads config for thresholds and max templates.
        """
        cfg = self.config.get('template_updating', {})
        if not cfg.get('enabled', False):
            console.print("[dim]Template updating disabled in config[/dim]")
            return {'success': False, 'reason': 'disabled'}

        min_sim = float(cfg.get('min_similarity', 0.8))
        min_snr = float(cfg.get('min_snr', 5.0))
        max_templates = int(cfg.get('max_templates', 500))

        # Check current template count
        existing_templates = list(templates_dir.glob("*.mseed"))
        remaining_slots = max(0, max_templates - len(existing_templates))
        if remaining_slots == 0:
            console.print(f"[yellow]Max templates ({max_templates}) already reached[/yellow]")
            return {'success': False, 'reason': 'max_templates_reached'}

        # Select candidates
        candidates = self.select_candidates(
            detections_df, min_similarity=min_sim, min_snr=min_snr,
            max_candidates=remaining_slots,
        )

        if len(candidates) == 0:
            console.print("[dim]No candidates met the criteria for template promotion[/dim]")
            return {'success': True, 'n_added': 0}

        # Load existing template info
        info_file = templates_dir / self.config.get('paths.template_info_file', 'template_info.csv')
        existing_info = pd.read_csv(info_file) if info_file.exists() else None

        # Extract and save
        result = self.extract_and_save_templates(candidates, templates_dir, existing_info)

        # Update catalog
        if result['n_added'] > 0:
            self.update_template_catalog(templates_dir, result['new_info_df'])

        return {
            'success': True,
            'n_added': result['n_added'],
        }
