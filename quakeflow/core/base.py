"""
Base classes and shared functionality.
"""

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream
from typing import Optional, Dict, Any, List
from pathlib import Path
from rich.console import Console

from ..config import Config

console = Console()


class BaseProcessor:
    """Base class for all processors with shared functionality."""
    
    def __init__(self, config: Config):
        self.config = config
        self._sq = None
        self._sds = None
        self._backend: Optional[str] = None  # 'squirrel' or 'sds'

    # ------------------------------------------------------------------
    # Waveform backend selection
    # ------------------------------------------------------------------

    @property
    def waveform_backend(self) -> str:
        """Return the active waveform backend name ('squirrel' or 'sds')."""
        if self._backend is None:
            sds_root = self.config.get('sds.root', None)
            if sds_root and str(sds_root).strip():
                self._backend = 'sds'
            else:
                self._backend = 'squirrel'
        return self._backend

    def get_squirrel(self):
        """Get or create Squirrel instance (only when backend is squirrel)."""
        if self._sq is None:
            from pyrocko import squirrel
            self._sq = squirrel.Squirrel(
                persistent=self.config['squirrel.persistent']
            )
        return self._sq

    def get_sds_client(self):
        """Get or create SDS waveform client."""
        if self._sds is None:
            from .sds_client import SDSWaveformClient
            sds_root = self.config.get('sds.root', '')
            sds_type = self.config.get('sds.type', 'D')
            cache_size = int(self.config.get('sds.cache_size', 64))
            max_workers = int(self.config.get('sds.max_workers', 8))
            fileborder = float(self.config.get('sds.fileborder_seconds', 30.0))
            self._sds = SDSWaveformClient(
                sds_root=sds_root,
                sds_type=sds_type,
                cache_size=cache_size,
                max_workers=max_workers,
                fileborder_seconds=fileborder,
            )
        return self._sds

    # ------------------------------------------------------------------
    # Multi-station helpers
    # ------------------------------------------------------------------
    def get_station_list(self) -> List[Dict[str, Any]]:
        """Return list of station definitions.

        If ``stations.networks`` is populated in config, use it.
        Otherwise fall back to the legacy single-station fields.
        Each entry has keys: code, network, channels, lat, lon, weight.
        """
        # Prefer properly nested stations.networks
        networks = self.config.get('stations.networks', [])
        # Also accept a top-level 'networks' key (common mis-indent in configs)
        if not networks:
            try:
                alt = self.config.get('networks', [])
                if isinstance(alt, list) and alt:
                    networks = alt
            except Exception:
                pass
        if networks:
            # Ensure every entry has defaults
            result = []
            for net in networks:
                entry = {
                    'code': net.get('code', self.config.get('stations.station_code', 'XXXX')),
                    'network': net.get('network', '*'),
                    'channels': net.get('channels', self.config.get('stations.channels', ['Z', 'N', 'E'])),
                    'lat': net.get('lat', self.config.get('stations.lat', 0.0)),
                    'lon': net.get('lon', self.config.get('stations.lon', 0.0)),
                    'weight': float(net.get('weight', 1.0)),
                }
                result.append(entry)
            return result
        # Legacy single-station config
        return [{
            'code': self.config.get('stations.station_code', 'XXXX'),
            'network': '*',
            'channels': self.config.get('stations.channels', ['Z', 'N', 'E']),
            'lat': self.config.get('stations.lat', 0.0),
            'lon': self.config.get('stations.lon', 0.0),
            'weight': 1.0,
        }]

    def get_primary_station(self) -> Dict[str, Any]:
        """Return the primary (first) station definition."""
        return self.get_station_list()[0]
    
    def load_waveforms_multi(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        stations: Optional[List[Dict]] = None,
    ) -> Dict[str, Stream]:
        """Load waveforms for multiple stations.

        Returns dict keyed by ``station_code`` → merged ObsPy Stream.
        """
        if stations is None:
            stations = self.get_station_list()
        result = {}
        for sdef in stations:
            code = sdef['code']
            channels = sdef.get('channels', ['Z', 'N', 'E'])
            # Load all channels at once using a broad wildcard and filter later
            st_all = Stream()
            for ch in channels:
                st_ch = self.load_waveforms(starttime, endtime, code, ch)
                if st_ch is not None:
                    st_all += st_ch
            if len(st_all) > 0:
                result[code] = st_all
        return result
    
    def load_catalog_events(self, catalog_path: Path, catalog_type: str = "generic") -> pd.DataFrame:
        """Load and parse catalog events.

        Supported `catalog_type` values:
        - ``generic``: CSV with a `time` column parsable by pandas (ISO-like strings).
        - ``grun``: Special GRUN-style catalog with timestamps embedded in a free-text field.
        - ``dlf``, ``dlff``, ``dat``: Whitespace-delimited (DLF-like) catalogs commonly
            distributed as `.dat` files. These expect columns typically found in regional
            catalogs (see handling below).

        The DLF/`.dat` handler accepts files with at least six columns (whitespace-delimited):
        0: date in `YYYYMMDD` (e.g. `20180105`)
        1: time in `HHMM` or `HMM` (leading zeros may be omitted; seconds are ignored)
        2: location/station or event id (ignored)
        3: magnitude
        4: latitude (decimal degrees)
        5: longitude (decimal degrees)
        6: depth (optional, km)

        Example DLF/.dat line:
                20180105  905  -  2.3  50.1234  7.9876  10.0

        Returns a DataFrame with at least `event_time`, `lat`, `lon`, and `magnitude`.
        """
        # --- Qseek run folder (may be a directory) ----------------------------
        if catalog_type == "qseek":
            return self._load_qseek_catalog(catalog_path)

        # Try to read CSV by default; certain special formats will override
        try:
            df = pd.read_csv(catalog_path)
        except Exception:
            # Fallback to whitespace-delimited if CSV read fails
            df = pd.read_csv(catalog_path, sep=r'\s+', header=None, comment='#')
        
        if catalog_type == "grun":
            import re
            def parse_grun_time(time_str):
                m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)", str(time_str))
                return UTCDateTime(m.group(1)) if m else None
            
            df["event_time"] = df["time"].apply(parse_grun_time)
            df = df[df["event_time"].notnull()]
            df["magnitude"] = df.get("Magnitude", np.nan)
            df["lat"] = df["latitude"].astype(float)
            df["lon"] = df["longitude"].astype(float)
            
        elif catalog_type == "generic":
            times = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df["event_time"] = times.apply(lambda t: UTCDateTime(t) if pd.notna(t) else None)
            df = df[df["event_time"].notnull()]
            df["magnitude"] = pd.to_numeric(df.get("ML-south-west-germany", np.nan), errors="coerce")
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df = df.dropna(subset=["lat", "lon"])
        

        # --- Custom DLF-like whitespace format
        if catalog_type in ("dlf", "dlff", "dat"):
            # Expect columns: date(yyyymmdd), time(hhmm possibly without leading zeros), location, magnitude, lat, lon, depth
            # Read as whitespace-delimited if not already
            try:
                df_raw = pd.read_csv(catalog_path, sep=r'\s+', header=None, comment='#')
            except Exception:
                df_raw = pd.read_csv(catalog_path, header=None, comment='#')

            # Ensure we have at least 6 columns
            if df_raw.shape[1] < 6:
                raise ValueError(f"Unexpected DLF catalog format, need at least 6 columns, got {df_raw.shape[1]}")

            # map columns
            # col0: date yyyymmdd, col1: time hhmm (may miss leading zeros), col2: location (ignored), col3: magnitude, col4: lat, col5: lon, col6: depth (optional)
            df = pd.DataFrame()
            date_col = df_raw.iloc[:, 0].astype(str).str.strip()
            time_col = df_raw.iloc[:, 1].astype(str).str.strip()

            # pad time to 4 characters (hhmm)
            time_col = time_col.apply(lambda s: s.zfill(4))

            # combine and parse
            dt_series = pd.to_datetime(date_col + time_col, format="%Y%m%d%H%M", errors='coerce')
            df['event_time'] = dt_series.apply(lambda t: UTCDateTime(t) if pd.notna(t) else None)

            # magnitude, lat, lon, depth
            df['magnitude'] = pd.to_numeric(df_raw.iloc[:, 3], errors='coerce')
            df['lat'] = pd.to_numeric(df_raw.iloc[:, 4], errors='coerce')
            df['lon'] = pd.to_numeric(df_raw.iloc[:, 5], errors='coerce')
            if df_raw.shape[1] > 6:
                df['depth'] = pd.to_numeric(df_raw.iloc[:, 6], errors='coerce')
            else:
                df['depth'] = np.nan

            # drop rows without valid times or coordinates
            df = df[df['event_time'].notnull()]
            df = df.dropna(subset=['lat', 'lon'])

        return df

    # ------------------------------------------------------------------
    # Qseek catalog loader
    # ------------------------------------------------------------------
    def _load_qseek_catalog(self, qseek_path: Path) -> pd.DataFrame:
        """Load a qseek detection catalog from a run folder or its CSV.

        Accepts either the qseek run directory (containing ``csv/detections.csv``,
        ``pyrocko_markers/``, etc.) or the CSV file itself.

        The returned DataFrame has: ``event_time``, ``lat``, ``lon``,
        ``depth``, ``magnitude`` (= semblance), ``semblance``, ``n_picks``,
        ``n_stations``, ``uncertainty_horizontal``, ``uncertainty_vertical``,
        and ``_marker_file`` (path to the per-event marker file, if found).

        Filtering is controlled by config keys under ``qseek_filter.*``:
        - ``min_semblance``    (default 0.3)
        - ``min_n_picks``      (default 6)
        - ``max_uncertainty_horizontal`` (default 500 m)
        """
        qseek_path = Path(qseek_path)

        # Accept either the directory or the CSV directly
        if qseek_path.is_dir():
            csv_file = qseek_path / "csv" / "detections.csv"
            markers_dir = qseek_path / "pyrocko_markers"
        else:
            csv_file = qseek_path
            markers_dir = qseek_path.parent.parent / "pyrocko_markers"

        if not csv_file.exists():
            raise FileNotFoundError(f"Qseek detections CSV not found: {csv_file}")

        df = pd.read_csv(csv_file)
        console.print(f"[cyan]📋 Loaded qseek catalog: {len(df)} detections[/cyan]")

        # Parse time → UTCDateTime
        times = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df["event_time"] = times.apply(lambda t: UTCDateTime(t) if pd.notna(t) else None)
        df = df[df["event_time"].notnull()]

        # Ensure numeric columns
        for col in ("lat", "lon", "depth", "semblance", "n_picks", "n_stations",
                     "uncertainty_horizontal", "uncertainty_vertical"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Use semblance as magnitude proxy
        df["magnitude"] = df.get("semblance", np.nan)

        # --- Quality filtering ------------------------------------------------
        min_sem = float(self.config.get("qseek_filter.min_semblance", 0.3))
        min_picks = int(self.config.get("qseek_filter.min_n_picks", 6))
        max_unc_h = float(self.config.get("qseek_filter.max_uncertainty_horizontal", 500.0))

        n_before = len(df)
        if "semblance" in df.columns:
            df = df[df["semblance"] >= min_sem]
        if "n_picks" in df.columns:
            df = df[df["n_picks"] >= min_picks]
        if "uncertainty_horizontal" in df.columns:
            df = df[df["uncertainty_horizontal"] <= max_unc_h]

        console.print(
            f"[cyan]  After quality filter: {len(df)}/{n_before} events "
            f"(semblance≥{min_sem}, n_picks≥{min_picks}, unc_h≤{max_unc_h}m)[/cyan]"
        )

        # --- Map per-event marker files so picks can be extracted later --------
        if markers_dir.is_dir():
            # Build a lookup from event time → marker file
            # Use float timestamps as keys (UTCDateTime.__hash__ can fail)
            marker_files = sorted(markers_dir.glob("*.list"))
            marker_map: list[tuple[float, Path]] = []
            for mf in marker_files:
                # Filename looks like: 2026-01-01T010149.360+0000.list
                try:
                    t = UTCDateTime(mf.stem)
                    marker_map.append((float(t), mf))
                except Exception:
                    continue

            def _find_marker(evt_time):
                if evt_time is None:
                    return None
                evt_ts = float(evt_time)
                for ts, path in marker_map:
                    if abs(ts - evt_ts) < 0.5:
                        return str(path)
                return None

            df["_marker_file"] = df["event_time"].apply(_find_marker)
            n_markers = df["_marker_file"].notna().sum()
            console.print(f"[cyan]  Matched {n_markers}/{len(df)} events to per-event marker files[/cyan]")
        else:
            df["_marker_file"] = None

        df = df.dropna(subset=["lat", "lon"])
        return df.reset_index(drop=True)
    
    def filter_by_bbox(self, df: pd.DataFrame, 
                      bbox: tuple) -> pd.DataFrame:
        """Filter events within bounding box."""
        min_lat, max_lat, min_lon, max_lon = bbox
        return df[
            (df["lat"] >= min_lat) & (df["lat"] <= max_lat) &
            (df["lon"] >= min_lon) & (df["lon"] <= max_lon)
        ]
    
    def filter_by_region(self, df: pd.DataFrame, region_name: str) -> pd.DataFrame:
        """
        Filter catalog by predefined region.
        
        Parameters:
        -----------
        region_name : str
            Name of predefined region
            
        Returns:
        --------
        CatalogManager : New manager with filtered data
        """
        # Define common regions (can be extended)
        # regions = {
        #     'germany_west': (5.0, 10.0, 47.0, 55.0),
        #     'germany_south': (7.0, 13.0, 47.0, 50.0),
        #     'germany_north': (5.0, 15.0, 50.0, 55.0),
        #     'rhenish_massif': (6.5, 8.5, 49.5, 51.5),
        #     'Moersdorf': (7.311, 7.3755, 50.0778, 50.1283),
        #     'Bad_Schwalbach': (7.8017, 8.0077, 49.994, 50.2027),
        #     'Ochtendung': (7.293, 7.483, 50.27, 50.411)
        # }
        regions = {
            'germany_west': (47.0, 55.0, 5.0, 10.0),
            'germany_south': (47.0, 50.0, 7.0, 13.0),
            'germany_north': (50.0, 55.0, 5.0, 15.0),
            'rhenish_massif': (49.5, 51.5, 6.5, 8.5),
            'Moersdorf': (50.0778, 50.1283, 7.311, 7.3755),
            'Bad_Schwalbach': (49.994, 50.2027, 7.8017, 8.0077),
            'Ochtendung': (50.27, 50.411, 7.293, 7.483)
        }
        if region_name not in regions:
            console.print(f"[red]Region '{region_name}' not recognized![/red]")
            return df
        bbox = regions[region_name]
        return self.filter_by_bbox(df, bbox)
    
    def load_waveforms(self, starttime, 
                    endtime,
                    station: str,
                    channel: str) -> Optional[Stream]:
        """Load waveforms for given time range with proper data type handling.

        Dispatches to the SDS client when ``sds.root`` is configured,
        otherwise falls back to Pyrocko Squirrel.
        """
        # Ensure UTCDateTime (callers may pass pandas Timestamp or datetime)
        if not isinstance(starttime, UTCDateTime):
            if hasattr(starttime, 'timestamp'):
                starttime = UTCDateTime(starttime.timestamp())
            else:
                starttime = UTCDateTime(str(starttime))
        if not isinstance(endtime, UTCDateTime):
            if hasattr(endtime, 'timestamp'):
                endtime = UTCDateTime(endtime.timestamp())
            else:
                endtime = UTCDateTime(str(endtime))
        if self.waveform_backend == 'sds':
            return self._load_waveforms_sds(starttime, endtime, station, channel)
        return self._load_waveforms_squirrel(starttime, endtime, station, channel)

    # ------------------------------------------------------------------
    # SDS backend
    # ------------------------------------------------------------------

    def _load_waveforms_sds(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        station: str,
        channel: str,
    ) -> Optional[Stream]:
        """Load waveforms from the SDS archive."""
        client = self.get_sds_client()
        st = client.get_waveforms(
            network="*",
            station=station,
            location="*",
            channel=f"*{channel}",
            starttime=starttime,
            endtime=endtime,
            merge=1,
        )
        if st is None or len(st) == 0:
            return None
        # Ensure float32
        for tr in st:
            if tr.data.dtype != np.float32:
                tr.data = tr.data.astype(np.float32)
        return st

    # ------------------------------------------------------------------
    # Squirrel backend
    # ------------------------------------------------------------------

    def _load_waveforms_squirrel(
        self,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        station: str,
        channel: str,
    ) -> Optional[Stream]:
        """Load waveforms via Pyrocko Squirrel."""
        from pyrocko import obspy_compat
        sq = self.get_squirrel()
        
        try:
            tmin = float(starttime.timestamp)
            tmax = float(endtime.timestamp)
        except TypeError:
            tmin = float(starttime.timestamp())
            tmax = float(endtime.timestamp())
        
        traces = sq.get_waveforms(
            codes=[f"*.{station}.*.*{channel}"],
            tmin=tmin,
            tmax=tmax
        )
        
        if not traces:
            return None
        
        st = Stream()
        for tr in traces:
            try:
                tr_obs = obspy_compat.to_obspy_trace(tr)
                
                # Convert to float32 immediately to avoid dtype issues
                if tr_obs.data.dtype != np.float32:
                    tr_obs.data = tr_obs.data.astype(np.float32)
                
                st.append(tr_obs)
            except Exception as e:
                console.print(f"[dim]Warning loading trace: {e}[/dim]")
                continue
        
        if not st:
            return None
        
        # Merge with proper handling for different dtypes
        try:
            for tr in st:
                if tr.data.dtype != np.float32:
                    tr.data = tr.data.astype(np.float32)
            st.merge(method=1, fill_value=0.0)
        except Exception as e:
            console.print(f"[yellow]Warning during merge: {e}[/yellow]")
            try:
                st = st.merge(method=0)
            except:
                if len(st) > 0:
                    st = Stream([st[0]])
                else:
                    return None
        
        return st
    
    def apply_bandpass_filter(self, stream: Stream, 
                             freqmin: float = 1.0, 
                             freqmax: float = 30.0) -> Stream:
        """Apply bandpass filter to stream."""
        for tr in stream:
            tr.filter(
                "bandpass",
                freqmin=freqmin,
                freqmax=freqmax,
                corners=4,
                zerophase=True
            )
        return stream