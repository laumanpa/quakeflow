"""
Base classes and shared functionality.
"""

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream
from pyrocko import squirrel, obspy_compat
from typing import Optional, Dict, Any
from pathlib import Path
from rich.console import Console

from ..config import Config

console = Console()


class BaseProcessor:
    """Base class for all processors with shared functionality."""
    
    def __init__(self, config: Config):
        self.config = config
        self._sq = None
    
    def get_squirrel(self):
        """Get or create Squirrel instance."""
        if self._sq is None:
            self._sq = squirrel.Squirrel(
                persistent=self.config['squirrel.persistent']
            )
        return self._sq
    
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
        # Try to read CSV by default; certain special formats will override
        try:
            df = pd.read_csv(catalog_path)
        except Exception:
            # Fallback to whitespace-delimited if CSV read fails
            df = pd.read_csv(catalog_path, delim_whitespace=True, header=None, comment='#')
        
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
                df_raw = pd.read_csv(catalog_path, delim_whitespace=True, header=None, comment='#')
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
    
    def load_waveforms(self, starttime: UTCDateTime, 
                    endtime: UTCDateTime,
                    station: str,
                    channel: str) -> Optional[Stream]:
        """Load waveforms for given time range with proper data type handling."""
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
            # Ensure all traces have the same dtype before merging
            for tr in st:
                if tr.data.dtype != np.float32:
                    tr.data = tr.data.astype(np.float32)
            
            # Use float32 fill value
            st.merge(method=1, fill_value=0.0)
            
        except Exception as e:
            console.print(f"[yellow]Warning during merge: {e}[/yellow]")
            # Try alternative merge method
            try:
                st = st.merge(method=0)  # Don't fill gaps
            except:
                # If still fails, just use the first trace
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