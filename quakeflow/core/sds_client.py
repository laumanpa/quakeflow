"""
Fast SDS (SeisComP Data Structure) waveform reader.

Provides a drop-in replacement for Pyrocko Squirrel waveform access that
reads directly from an SDS directory tree::

    <root>/<year>/<net>/<sta>/<cha>.<type>/<net>.<sta>.<loc>.<cha>.<type>.<year>.<jday>

This uses ObsPy's ``obspy.clients.filesystem.sds.Client`` under the hood
and wraps it with:

* **Thread-parallel I/O** – reads spanning multiple day-files are loaded
  concurrently via a ``ThreadPoolExecutor``.
* **LRU day-file cache** – recently read day-files are kept in memory to
  avoid redundant disk reads for neighbouring events.

Usage::

    client = SDSWaveformClient("/data/sds", cache_size=64, max_workers=8)
    st = client.get_waveforms("*", "MOED", "*", "*Z", t1, t2)
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
from obspy import Stream, UTCDateTime, read

__all__ = ["SDSWaveformClient"]


class _LRUCache:
    """Simple thread-safe LRU cache for ObsPy Streams keyed by file path."""

    def __init__(self, maxsize: int = 64):
        self._maxsize = max(1, maxsize)
        self._cache: OrderedDict[str, Stream] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Stream]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key].copy()
            return None

    def put(self, key: str, value: Stream) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value.copy()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class SDSWaveformClient:
    """High-performance SDS archive reader.

    Parameters
    ----------
    sds_root : str or Path
        Root directory of the SDS archive.
    sds_type : str
        SDS data type character (default ``'D'`` for waveform data).
    cache_size : int
        Maximum number of day-files to keep in the LRU cache.
    max_workers : int
        Thread-pool size for parallel day-file reads.
    fileborder_seconds : float
        Extra seconds to load from neighbouring day-files at day boundaries.
    """

    def __init__(
        self,
        sds_root: str | Path,
        sds_type: str = "D",
        cache_size: int = 64,
        max_workers: int = 8,
        fileborder_seconds: float = 30.0,
    ):
        self.sds_root = Path(sds_root)
        if not self.sds_root.is_dir():
            raise FileNotFoundError(f"SDS root directory not found: {self.sds_root}")
        self.sds_type = sds_type
        self.cache = _LRUCache(cache_size)
        self.max_workers = max(1, max_workers)
        self.fileborder_seconds = fileborder_seconds

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _sds_path(
        root: Path, year: int, jday: int,
        net: str, sta: str, loc: str, cha: str, sds_type: str,
    ) -> Path:
        """Build the canonical SDS file path.

        Tries zero-padded (``037``) first, falls back to unpadded (``37``)
        since SDS archives vary in their convention.
        """
        base = root / str(year) / net / sta / f"{cha}.{sds_type}"
        padded = base / f"{net}.{sta}.{loc}.{cha}.{sds_type}.{year}.{jday:03d}"
        if padded.exists():
            return padded
        unpadded = base / f"{net}.{sta}.{loc}.{cha}.{sds_type}.{year}.{jday}"
        if unpadded.exists():
            return unpadded
        return padded  # return canonical path even if missing

    def _resolve_files(
        self,
        network: str, station: str, location: str, channel: str,
        starttime: UTCDateTime, endtime: UTCDateTime,
    ) -> List[Path]:
        """Enumerate all SDS day-files that cover ``[starttime, endtime]``.

        Supports simple wildcards (``*``, ``?``) in *network*, *station*,
        *location*, and *channel* by scanning the actual directory tree.
        """
        has_wildcard = any("*" in s or "?" in s for s in (network, station, location, channel))

        # Determine the range of Julian days to cover
        t = starttime - self.fileborder_seconds
        t_end = endtime + self.fileborder_seconds
        days: list[tuple[int, int]] = []
        while t <= t_end:
            days.append((t.year, t.julday))
            t += 86400

        if not has_wildcard:
            # Fast path: directly construct paths
            return [
                self._sds_path(
                    self.sds_root, year, jday,
                    network, station, location, channel, self.sds_type,
                )
                for year, jday in days
            ]

        # Wildcard path: scan directories for matching entries
        files: list[Path] = []
        for year, jday in days:
            year_dir = self.sds_root / str(year)
            if not year_dir.is_dir():
                continue
            for net_dir in year_dir.iterdir():
                if not net_dir.is_dir() or not fnmatch.fnmatch(net_dir.name, network):
                    continue
                for sta_dir in net_dir.iterdir():
                    if not sta_dir.is_dir() or not fnmatch.fnmatch(sta_dir.name, station):
                        continue
                    for cha_dir in sta_dir.iterdir():
                        if not cha_dir.is_dir():
                            continue
                        cha_name = cha_dir.name.split(".")[0] if "." in cha_dir.name else cha_dir.name
                        if not fnmatch.fnmatch(cha_name, channel):
                            continue
                        # Scan for matching day-files (both padded and unpadded jday)
                        jday_padded = f"{jday:03d}"
                        jday_unpadded = str(jday)
                        for fpath in cha_dir.iterdir():
                            fname = fpath.name
                            # Match: *.TYPE.YEAR.JDAY (either padding)
                            if not (fname.endswith(f".{self.sds_type}.{year}.{jday_padded}")
                                    or fname.endswith(f".{self.sds_type}.{year}.{jday_unpadded}")):
                                continue
                            if True:  # formerly: fnmatch.fnmatch(fpath.name, pattern)
                                # Optionally filter by location in filename
                                parts = fpath.name.split(".")
                                if len(parts) >= 4:
                                    file_loc = parts[2]
                                    if not fnmatch.fnmatch(file_loc, location):
                                        continue
                                files.append(fpath)
        return files

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read_file(self, path: Path) -> Stream:
        """Read a single day-file, using cache if available."""
        key = str(path)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if not path.exists():
            return Stream()

        try:
            st = read(str(path))
            # Convert to float32 early to save memory
            for tr in st:
                if tr.data.dtype != np.float32:
                    tr.data = tr.data.astype(np.float32)
            self.cache.put(key, st)
            return st.copy()
        except Exception:
            return Stream()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_waveforms(
        self,
        network: str = "*",
        station: str = "*",
        location: str = "*",
        channel: str = "*",
        starttime: UTCDateTime = None,
        endtime: UTCDateTime = None,
        merge: int = 1,
    ) -> Stream:
        """Load waveforms from the SDS archive.

        Signature mirrors ``obspy.clients.filesystem.sds.Client.get_waveforms``
        but adds parallel I/O and caching.

        Parameters
        ----------
        network, station, location, channel : str
            SEED codes, may contain ``*`` / ``?`` wildcards.
        starttime, endtime : UTCDateTime
            Requested time window.
        merge : int
            ObsPy merge method (default 1 = fill gaps with zeros).
            Use ``-1`` to skip merging.

        Returns
        -------
        Stream
            Trimmed & merged ObsPy Stream.
        """
        if starttime is None or endtime is None:
            return Stream()

        files = self._resolve_files(network, station, location, channel, starttime, endtime)
        if not files:
            return Stream()

        # Parallel read
        st = Stream()
        if len(files) == 1:
            st = self._read_file(files[0])
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as pool:
                futures = {pool.submit(self._read_file, f): f for f in files}
                for fut in as_completed(futures):
                    try:
                        st += fut.result()
                    except Exception:
                        continue

        if len(st) == 0:
            return st

        # Trim to exact window
        st.trim(starttime, endtime)

        # Merge
        if merge >= 0 and len(st) > 1:
            try:
                st.merge(method=merge, fill_value=0)
            except Exception:
                pass

        return st

    def get_waveforms_bulk(
        self,
        bulk: List[tuple],
        merge: int = 1,
    ) -> Stream:
        """Load many time windows in parallel.

        Parameters
        ----------
        bulk : list of (network, station, location, channel, starttime, endtime)
            Each tuple describes one waveform request.
        merge : int
            ObsPy merge method applied per-request.

        Returns
        -------
        Stream
            Combined stream for all requests.
        """
        result = Stream()

        def _fetch(req):
            net, sta, loc, cha, t1, t2 = req
            return self.get_waveforms(net, sta, loc, cha, t1, t2, merge=merge)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_fetch, r): r for r in bulk}
            for fut in as_completed(futures):
                try:
                    result += fut.result()
                except Exception:
                    continue
        return result

    def clear_cache(self) -> None:
        """Drop all cached day-files."""
        self.cache.clear()
