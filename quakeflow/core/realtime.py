"""
Real-time / incremental processing mode.

Supports:
- State persistence: track which days have been processed
- Incremental updates: only process new data since last run
- Configurable chunk sizes for near-real-time operation
- Optional alert callbacks for significant detections
"""

import gc
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from obspy import UTCDateTime
from rich.console import Console

from .base import BaseProcessor
from .template_matcher import TemplateMatcher
from ..config import Config

console = Console()


class RealtimeProcessor(BaseProcessor):
    """Incremental / real-time template matching processor."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.state_file = Path(
            config.get('realtime.state_file', 'quakeflow_state.json')
        )
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load processing state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: could not load state file: {e}[/yellow]")
        return {
            'last_processed_date': None,
            'total_days_processed': 0,
            'total_detections': 0,
            'processing_history': [],
        }

    def _save_state(self):
        """Persist processing state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def get_unprocessed_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DatetimeIndex:
        """Determine which dates still need processing.

        If no start_date given, resumes from state['last_processed_date'].
        If no end_date given, goes up to yesterday.
        """
        if start_date is None:
            last = self.state.get('last_processed_date')
            if last is not None:
                start = pd.Timestamp(last) + pd.Timedelta(days=1)
            else:
                start = pd.Timestamp(self.config.get('template_matching.start_date', '2018-01-01'))
        else:
            start = pd.Timestamp(start_date)

        if end_date is None:
            end = pd.Timestamp(datetime.utcnow().date()) - pd.Timedelta(days=1)
        else:
            end = pd.Timestamp(end_date)

        if start > end:
            return pd.DatetimeIndex([])

        return pd.date_range(start=start, end=end, freq='D')

    def process_incremental(
        self,
        templates_dir: Path,
        dates: Optional[pd.DatetimeIndex] = None,
        alert_callback: Optional[Callable] = None,
    ) -> Dict:
        """Process only unprocessed dates incrementally.

        Parameters
        ----------
        templates_dir : directory containing templates
        dates : specific dates to process (defaults to unprocessed range)
        alert_callback : optional function(detection_dict) called for high-CC detections

        Returns
        -------
        dict with processing summary
        """
        if dates is None:
            dates = self.get_unprocessed_dates()

        if len(dates) == 0:
            console.print("[green]No new dates to process - up to date![/green]")
            return {'success': True, 'days_processed': 0, 'new_detections': 0}

        console.print(f"[cyan]📡 Real-time mode: processing {len(dates)} new days[/cyan]")
        console.print(f"   Range: {dates[0].date()} to {dates[-1].date()}")

        # Create matcher and run
        matcher = TemplateMatcher(self.config)
        result = matcher.match_templates(
            templates_dir,
        )

        # The matcher processes its own date range from config.
        # For true incremental, we override by processing specific dates.
        station = self.config.get('stations.station_code', 'XXXX')
        channel = self.config.get('stations.primary_channel', 'Z')
        output_dir = (
            self.config.get_path('base_dir') / "similarity" / f"{station}_{channel}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load templates
        template_files = sorted(templates_dir.glob("*.mseed"))
        if not template_files:
            console.print("[red]No templates found![/red]")
            return {'success': False}

        from obspy import read as obspy_read
        template_streams = []
        for f in template_files:
            st = obspy_read(str(f))
            st = matcher.apply_bandpass_filter(
                st,
                self.config.get('template_creation.filter_min', 1.0),
                self.config.get('template_creation.filter_max', 30.0),
            )
            # Store only a lightweight copy (data is float32 from base.py)
            template_streams.append([st.copy()])
            del st

        # Cluster if enabled
        cluster_enabled = bool(self.config.get('template_matching.cluster_enabled', True))
        if cluster_enabled:
            template_streams, repr_map = matcher.cluster_templates(
                template_streams,
                eps=self.config.get('template_matching.cluster_eps', 0.2),
            )

        # Process each date
        n_successful = 0
        total_new_dets = 0
        alert_threshold = float(self.config.get('realtime.alert_threshold', 0.8))
        all_stations = self.get_station_list()
        extra_stations = [s for s in all_stations if s['code'] != station] if len(all_stations) > 1 else None

        for date in dates:
            success = matcher.process_one_day(
                date, station, channel,
                template_streams,
                self.config.get('template_matching.similarity_threshold', 0.5),
                self.config.get('template_matching.distance_samples', 200),
                output_dir,
                extra_stations=extra_stations,
            )
            if success:
                n_successful += 1
                # Count new detections and check for alerts
                det_file = output_dir / f"detections_{date.strftime('%Y%m%d')}.csv"
                if det_file.exists():
                    try:
                        day_dets = pd.read_csv(det_file)
                        total_new_dets += len(day_dets)

                        # Alert for high-CC detections
                        if alert_callback is not None:
                            high_cc = day_dets[day_dets['similarity'] >= alert_threshold]
                            for _, det_row in high_cc.iterrows():
                                alert_callback(det_row.to_dict())
                    except Exception:
                        pass

            # Update state after each day
            self.state['last_processed_date'] = str(date.date())
            self.state['total_days_processed'] += 1 if success else 0
            self.state['total_detections'] += total_new_dets
            self.state['processing_history'].append({
                'date': str(date.date()),
                'success': success,
                'timestamp': datetime.utcnow().isoformat(),
            })
            self._save_state()
            gc.collect()  # free per-day memory between iterations

        console.print(f"\n✅ Incremental processing complete: "
                       f"{n_successful}/{len(dates)} days, {total_new_dets} new detections")

        return {
            'success': True,
            'days_processed': len(dates),
            'successful_days': n_successful,
            'new_detections': total_new_dets,
        }

    def get_status(self) -> Dict:
        """Return current processing status."""
        return {
            'last_processed': self.state.get('last_processed_date'),
            'total_days': self.state.get('total_days_processed', 0),
            'total_detections': self.state.get('total_detections', 0),
            'state_file': str(self.state_file),
        }
