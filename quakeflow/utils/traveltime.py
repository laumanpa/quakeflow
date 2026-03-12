"""
P-wave travel time estimation using pyrocko cake.

Computes first-arriving P-wave travel times from a 1D velocity model
for local/regional earthquakes.  Used to shift template windows from
origin time to estimated P-arrival time when no phase picks are available.
"""

import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers (pyrocko may not always be installed)
# ---------------------------------------------------------------------------
_cake = None
_orthodrome = None


def _ensure_pyrocko():
    """Import pyrocko modules on first use."""
    global _cake, _orthodrome
    if _cake is None:
        from pyrocko import cake as _c, orthodrome as _o
        _cake = _c
        _orthodrome = _o


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
_model_cache: dict = {}


def _load_model(velocity_model_path: Optional[str] = None):
    """Load and cache a pyrocko cake 1D velocity model.

    Parameters
    ----------
    velocity_model_path : str or None
        Path to a custom ``.nd`` or ``.tvel`` velocity model file.
        If *None* or empty, the pyrocko default model (ak135) is used.
    """
    _ensure_pyrocko()
    key = str(velocity_model_path) if velocity_model_path else "__default__"
    if key not in _model_cache:
        if velocity_model_path and Path(velocity_model_path).exists():
            _model_cache[key] = _cake.load_model(str(velocity_model_path))
            logger.info("Loaded custom velocity model: %s", velocity_model_path)
        else:
            _model_cache[key] = _cake.load_model()  # ak135
            logger.info("Using default velocity model (ak135)")
    return _model_cache[key]


# ---------------------------------------------------------------------------
# Distance helper
# ---------------------------------------------------------------------------

def compute_distance_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Compute surface distance in km between two geographic points.

    Uses pyrocko's high-accuracy Vincenty implementation.
    """
    _ensure_pyrocko()
    dist_m = _orthodrome.distance_accurate50m(lat1, lon1, lat2, lon2)
    return float(dist_m) / 1000.0


# ---------------------------------------------------------------------------
# Core travel-time computation
# ---------------------------------------------------------------------------

def compute_p_traveltime(
    source_lat: float,
    source_lon: float,
    source_depth_m: float,
    station_lat: float,
    station_lon: float,
    velocity_model_path: Optional[str] = None,
) -> float:
    """Estimate first-arriving P-wave travel time (seconds).

    For local/regional events the function:
    1. Computes epicentral distance and hypocentral distance.
    2. First tries pyrocko cake ray tracing with multiple crustal phases
       (``p``, ``P``, ``p(moho)P``, ``P(moho)p``).
    3. If no ray-traced arrival is found (common at very short distances in
       homogeneous layers), falls back to a straight-ray integration through
       the 1D velocity model layers.

    Parameters
    ----------
    source_lat, source_lon : float
        Earthquake epicenter (degrees).
    source_depth_m : float
        Source depth in **metres** (positive downward).
    station_lat, station_lon : float
        Station coordinates (degrees).
    velocity_model_path : str or None
        Path to a custom velocity model file.  ``None`` → ak135.

    Returns
    -------
    float
        Estimated P-wave travel time in seconds.
        Returns 0.0 if source and receiver are co-located.
    """
    _ensure_pyrocko()

    model = _load_model(velocity_model_path)

    # Epicentral distance (km) and hypocentral distance
    dist_km = compute_distance_km(source_lat, source_lon, station_lat, station_lon)
    depth_km = max(source_depth_m / 1000.0, 0.0)
    d_hypo_km = math.sqrt(dist_km ** 2 + depth_km ** 2)

    if d_hypo_km < 0.001:
        return 0.0

    # ------------------------------------------------------------------
    # 1) Try pyrocko cake ray tracing with all classic P phases
    # ------------------------------------------------------------------
    dist_deg = dist_km / 111.19
    dist_rad = dist_deg * _cake.d2r

    phases = list(_cake.PhaseDef.classic("P"))
    try:
        arrivals = model.arrivals(
            distances=[dist_rad],
            phases=phases,
            zstart=source_depth_m,
        )
    except Exception:
        arrivals = []

    if arrivals:
        arrivals.sort(key=lambda a: a.t)
        t_ray = arrivals[0].t
        if t_ray > 0.0:
            return float(t_ray)

    # ------------------------------------------------------------------
    # 2) Fallback: straight-ray travel time through the 1D model
    #    Very accurate for local events in the upper crust.
    # ------------------------------------------------------------------
    return _straight_ray_traveltime(model, source_depth_m, dist_km * 1000.0)


def _straight_ray_traveltime(model, source_depth_m: float, distance_m: float) -> float:
    """Compute travel time along a straight ray from source to surface receiver.

    Integrates P-wave velocity along the straight-line path through all
    model layers between source depth and the surface.

    Parameters
    ----------
    model : pyrocko cake LayeredModel
        1D velocity model.
    source_depth_m : float
        Source depth in metres.
    distance_m : float
        Epicentral (surface) distance in metres.
    """
    d_hypo = math.sqrt(distance_m ** 2 + source_depth_m ** 2)
    if d_hypo < 1.0:
        return 0.0

    # Sample depths along the straight ray (finer near interfaces)
    n_samples = max(200, int(d_hypo / 5.0))  # every ~5 m, min 200
    depths = np.linspace(source_depth_m, 0.0, n_samples)

    # Get Vp at each sampled depth
    vp = np.empty(n_samples)
    for i, z in enumerate(depths):
        try:
            mat = model.material(z)
            vp[i] = mat.vp
        except Exception:
            # Outside model range — use shallowest/deepest Vp
            vp[i] = vp[max(0, i - 1)] if i > 0 else 5800.0

    # Average Vp along path
    vp_avg = np.mean(vp)
    if vp_avg <= 0.0:
        vp_avg = 5800.0  # safe fallback

    return d_hypo / vp_avg


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def compute_p_traveltimes_batch(
    events,
    station_lat: float,
    station_lon: float,
    velocity_model_path: Optional[str] = None,
    depth_col: str = "depth",
    default_depth_m: float = 8000.0,
) -> np.ndarray:
    """Compute P travel times for a batch of events.

    Parameters
    ----------
    events : DataFrame
        Must have ``lat`` and ``lon`` columns, and optionally *depth_col*.
    station_lat, station_lon : float
        Station coordinates.
    velocity_model_path : str or None
        Custom velocity model path.
    depth_col : str
        Column name for source depth (in metres).
    default_depth_m : float
        Default depth when the depth column is missing or NaN.

    Returns
    -------
    np.ndarray
        Array of P travel times (seconds), one per event row.
    """
    n = len(events)
    tt = np.zeros(n)

    for i, (_, row) in enumerate(events.iterrows()):
        lat = float(row["lat"])
        lon = float(row["lon"])
        depth = float(row.get(depth_col, default_depth_m))
        if np.isnan(depth) or depth <= 0:
            depth = default_depth_m

        tt[i] = compute_p_traveltime(
            lat, lon, depth,
            station_lat, station_lon,
            velocity_model_path,
        )

    return tt
