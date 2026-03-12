"""
Evaluation of template matching results with robust magnitude estimation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from obspy.geodetics import gps2dist_azimuth
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor

from .base import BaseProcessor
from ..config import Config
from ..utils.plotting import (
    plot_magnitude_vs_time,
    plot_cumulative_events,
    plot_frequency_magnitude,
    plot_template_regression,
    qc_template_vs_detected,
    qc_residual_vs_similarity,
    qc_magnitude_vs_similarity,
    qc_residual_histogram,
    qc_time_magnitude,
    plot_waveform_comparison,
    plot_catalog_comparison,
    plot_template_redetection,
    plot_mw_vs_estimated,
    name2time
)

console = Console()


class ResultsEvaluator(BaseProcessor):
    """Evaluate template matching results and estimate magnitudes."""

    def _normalize_mapping_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize columns after merges with mapping to avoid _x/_y conflicts.

        - Ensures a single 'original_template_id' column exists
        - Drops any 'representative_id' variants (not needed post-merge)
        """
        # Promote suffixed original_template_id to canonical column
        if "original_template_id" not in df.columns:
            src = None
            if "original_template_id_y" in df.columns:
                src = df["original_template_id_y"]
            elif "original_template_id_x" in df.columns:
                src = df["original_template_id_x"]
            if src is not None:
                df["original_template_id"] = src

        # If both x/y exist, coalesce to a single column
        if "original_template_id_x" in df.columns and "original_template_id" in df.columns:
            df["original_template_id"] = df["original_template_id"].fillna(df["original_template_id_x"])

        # Drop any representative_id variants
        for c in ("representative_id", "representative_id_x", "representative_id_y"):
            if c in df.columns:
                df = df.drop(columns=[c])

        # Drop suffixed original_template_id columns
        for c in ("original_template_id_x", "original_template_id_y"):
            if c in df.columns:
                df = df.drop(columns=[c])

        return df

    def _load_mapping_and_merge(self, df: pd.DataFrame) -> tuple:
        """Load template ID mapping file and merge into detections DataFrame.
        
        Returns (df, merge_on) where merge_on is the column name to use
        when merging with template_info ('original_template_id' if mapping
        exists, otherwise 'template_id').
        """
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        merge_on = "template_id"
        if not mapping_file.exists():
            return df, merge_on
        try:
            if mapping_file.stat().st_size == 0:
                console.print(f"[yellow]Warning: mapping file {mapping_file} is empty — skipping mapping.[/yellow]")
                return df, merge_on
            mapping_df = pd.read_csv(mapping_file)
            if mapping_df.empty or not {"representative_id", "original_template_id"}.issubset(mapping_df.columns):
                console.print(f"[yellow]Warning: mapping file {mapping_file} missing required columns — skipping mapping.[/yellow]")
                return df, merge_on
            df = df.merge(
                mapping_df[["representative_id", "original_template_id"]],
                left_on="template_id",
                right_on="representative_id",
                how="left"
            )
            df = self._normalize_mapping_columns(df)
            merge_on = "original_template_id"
            if "representative_id" in df.columns:
                df = df.drop(columns=["representative_id"])
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            console.print(f"[yellow]Warning: failed to read mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: unexpected error reading mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
        return df, merge_on

    # ---------------------------------------------------------------------
    # b-value
    # ---------------------------------------------------------------------
    def _estimate_mc_maxcurvature(self, mags: np.ndarray, bin_width: float = 0.1) -> float:
        """Estimate completeness magnitude using maximum curvature method.

        Mc is the magnitude bin with the highest frequency of events
        (the mode of the non-cumulative frequency-magnitude distribution).
        A correction of +0.2 is commonly applied (Woessner & Wiemer, 2005),
        but is capped so that Mc never exceeds the 90th percentile of the
        data — this prevents the correction from overshooting the catalogue
        range for narrow magnitude distributions typical of template matching.
        """
        mags = mags[np.isfinite(mags)]
        if len(mags) < 5:
            return np.nan
        bins = np.arange(
            np.floor(mags.min() * 10) / 10,
            np.ceil(mags.max() * 10) / 10 + bin_width,
            bin_width,
        )
        counts, edges = np.histogram(mags, bins=bins)
        if len(counts) == 0:
            return np.nan
        peak_idx = np.argmax(counts)
        mc_raw = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
        # Cap the +0.2 correction so Mc stays below the 75th-percentile
        # magnitude — ensures ≥25 % of events remain above Mc for a
        # stable b-value estimate.
        mc_corrected = mc_raw + 0.2
        mc_cap = np.percentile(mags, 75)
        mc = min(mc_corrected, mc_cap)
        return mc

    def calc_bvalue(
        self, mags: np.ndarray, Mc: Optional[float] = None
    ) -> Dict[str, float]:
        """Maximum likelihood b-value estimate (Aki, 1965).

        If Mc is not provided, it is estimated using the method configured
        in evaluation.mc_method (default: maxcurvature).

        Robust against narrow magnitude ranges: if fewer than 5 events lie
        above the estimated Mc the method retries with Mc = mode (no
        correction), then with Mc = median.
        """
        mags = np.asarray(mags, dtype=np.float64)
        mags = mags[np.isfinite(mags)]
        if len(mags) < 5:
            return {"Mc": np.nan, "b": np.nan, "a": np.nan}

        bin_width = 0.1

        def _bvalue(m, mc):
            above = m[m >= mc]
            if len(above) < 5:
                return None
            mean_mag = above.mean()
            denom = mean_mag - (mc - bin_width / 2.0)
            if denom <= 0.02:
                # mean ≈ Mc → b numerically unstable (would be > ~20)
                return None
            b = np.log10(np.e) / denom
            a = np.log10(len(above)) + b * mc
            return {"Mc": float(mc), "b": float(b), "a": float(a)}

        # --- Primary Mc estimation ---
        if Mc is None:
            mc_method = str(self.config.get("evaluation.mc_method", "maxcurvature")).lower()
            if mc_method == "maxcurvature":
                Mc = self._estimate_mc_maxcurvature(mags, bin_width=bin_width)
            else:
                Mc = np.percentile(mags, 90) - 0.1

        if np.isfinite(Mc):
            result = _bvalue(mags, Mc)
            if result is not None:
                return result

        # --- Fallbacks for narrow catalogues ---
        # 1) Mode without +0.2 correction
        bins = np.arange(
            np.floor(mags.min() * 10) / 10,
            np.ceil(mags.max() * 10) / 10 + bin_width,
            bin_width,
        )
        counts, edges = np.histogram(mags, bins=bins)
        if len(counts) > 0:
            peak_idx = int(np.argmax(counts))
            mc_raw = (edges[peak_idx] + edges[peak_idx + 1]) / 2.0
            result = _bvalue(mags, mc_raw)
            if result is not None:
                console.print(f"[yellow]b-value: using uncorrected Mc={mc_raw:.2f} (corrected Mc overshot catalogue range)[/yellow]")
                return result

        # 2) Median magnitude
        mc_median = float(np.median(mags))
        result = _bvalue(mags, mc_median)
        if result is not None:
            console.print(f"[yellow]b-value: using median Mc={mc_median:.2f} as fallback[/yellow]")
            return result

        # 3) Minimum magnitude (last resort – uses all data)
        mc_min = float(mags.min())
        result = _bvalue(mags, mc_min)
        if result is not None:
            console.print(f"[yellow]b-value: using minimum Mc={mc_min:.2f} as last resort[/yellow]")
            return result

        return {"Mc": np.nan, "b": np.nan, "a": np.nan}
    # ---------------------------------------------------------------------
    # Distance correction
    # ---------------------------------------------------------------------
    def _distance_correct_amplitude(
        self,
        amplitude: float,
        ev_lat: float,
        ev_lon: float,
        gamma: float
    ) -> float:
        """Geometrical spreading correction (scalar version)."""
        try:
            amplitude = float(amplitude)
            ev_lat = float(ev_lat)
            ev_lon = float(ev_lon)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(amplitude) or not np.isfinite(ev_lat) or not np.isfinite(ev_lon):
            return np.nan

        st_lat = self.config["stations.lat"]
        st_lon = self.config["stations.lon"]
        R0 = self.config["evaluation.reference_distance"]

        dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
        R = max(dist_m / 1000.0, R0)

        return amplitude * (R / R0) ** gamma

    def _distance_correct_amplitude_vec(
        self,
        amplitudes: np.ndarray,
        ev_lats: np.ndarray,
        ev_lons: np.ndarray,
        gamma: float,
    ) -> np.ndarray:
        """Vectorized geometrical spreading correction.

        Computes gps2dist_azimuth once per unique (lat, lon) pair and maps
        the result back to all rows, avoiding redundant geodesic computations.
        """
        st_lat = float(self.config["stations.lat"])
        st_lon = float(self.config["stations.lon"])
        R0 = float(self.config["evaluation.reference_distance"])

        def _to_float_array(arr):
            """Safely convert any input to a 1-D float64 array."""
            s = pd.Series(arr).reset_index(drop=True)
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

        amplitudes = _to_float_array(amplitudes)
        ev_lats = _to_float_array(ev_lats)
        ev_lons = _to_float_array(ev_lons)

        result = np.full(len(amplitudes), np.nan)
        valid = np.isfinite(amplitudes) & np.isfinite(ev_lats) & np.isfinite(ev_lons)
        if not valid.any():
            return result

        # Build lookup for unique (lat, lon) → distance_km
        coords = np.column_stack([ev_lats[valid], ev_lons[valid]])
        # Round to ~11 m precision to increase cache hits
        coords_key = np.round(coords, 4)
        unique_coords, inv_idx = np.unique(coords_key, axis=0, return_inverse=True)

        dist_km = np.empty(len(unique_coords))
        for i, (lat, lon) in enumerate(unique_coords):
            d_m, _, _ = gps2dist_azimuth(lat, lon, st_lat, st_lon)
            dist_km[i] = max(d_m / 1000.0, R0)

        # Map back
        R_all = dist_km[inv_idx]
        result[valid] = amplitudes[valid] * (R_all / R0) ** gamma
        return result

    def _compute_distances_vec(
        self,
        ev_lats: np.ndarray,
        ev_lons: np.ndarray,
    ) -> np.ndarray:
        """Vectorized distance calculation to station (km).

        Computes gps2dist_azimuth once per unique (lat, lon) pair.
        """
        st_lat = float(self.config["stations.lat"])
        st_lon = float(self.config["stations.lon"])

        def _to_float_array(arr):
            s = pd.Series(arr).reset_index(drop=True)
            return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

        ev_lats = _to_float_array(ev_lats)
        ev_lons = _to_float_array(ev_lons)

        result = np.full(len(ev_lats), np.nan)
        valid = np.isfinite(ev_lats) & np.isfinite(ev_lons)
        if not valid.any():
            return result

        coords = np.column_stack([ev_lats[valid], ev_lons[valid]])
        coords_key = np.round(coords, 4)
        unique_coords, inv_idx = np.unique(coords_key, axis=0, return_inverse=True)

        dist_km = np.empty(len(unique_coords))
        for i, (lat, lon) in enumerate(unique_coords):
            d_m, _, _ = gps2dist_azimuth(lat, lon, st_lat, st_lon)
            dist_km[i] = d_m / 1000.0

        result[valid] = dist_km[inv_idx]
        return result

    # ---------------------------------------------------------------------
    # Magnitude estimation (core change!)
    # ---------------------------------------------------------------------
    def estimate_magnitudes(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Estimate magnitudes using amplitude ratios:
        M_det = M_tpl + log10(A_det / A_tpl)
        """

        df = detections_df.copy()
        
        console.print(f"[dim]TEMPLATE MAGNITUDE DEBUG:[/dim]")
        console.print(f"[dim]  Template info shape: {template_info_df.shape}[/dim]")
        if 'magnitude' in template_info_df.columns:
            console.print(f"[dim]  Template magnitude range: {template_info_df['magnitude'].min():.2f} to {template_info_df['magnitude'].max():.2f}[/dim]")
        if 'amplitude' in template_info_df.columns:
            console.print(f"[dim]  Template amplitude range: {template_info_df['amplitude'].min():.2e} to {template_info_df['amplitude'].max():.2e}[/dim]")
        
        # Load mapping and determine merge key
        df, merge_on = self._load_mapping_and_merge(df)
        
        # Merge with template info and prefix template fields
        tpl_cols = [
            "amplitude",
            "amplitude_max",
            "amplitude_rms",
            "amplitude_percentile",
            "amplitude_envelope",
            "magnitude",
            "lat",
            "lon",
        ]
        df = df.merge(
            template_info_df[["template_id"] + [c for c in tpl_cols if c in template_info_df.columns]],
            left_on=merge_on,
            right_on="template_id",
            how="left",
            suffixes=("", "_tpl")
        )
        # Remove duplicated right key column
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])
        # Standardize template columns with tpl_ prefix
        for c in tpl_cols:
            if c in df.columns:
                df.rename(columns={c: f"tpl_{c}"}, inplace=True)

        # --- config
        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)

        # --- vectorized distance correction for template amplitudes
        tpl_lat_arr = df["tpl_lat"].values if "tpl_lat" in df.columns else np.full(len(df), np.nan)
        tpl_lon_arr = df["tpl_lon"].values if "tpl_lon" in df.columns else np.full(len(df), np.nan)

        df["tpl_amp_corr"] = self._distance_correct_amplitude_vec(
            df["tpl_amplitude"].values if "tpl_amplitude" in df.columns else np.full(len(df), np.nan),
            tpl_lat_arr, tpl_lon_arr, gamma,
        )

        # --- vectorized distance correction for detected event amplitudes
        # Use actual event coordinates when available (lat/lon or event_lat/event_lon),
        # otherwise fall back to template coords to preserve previous behaviour.
        ev_lat_arr = (
            df["lat"].values
            if "lat" in df.columns
            else (df["event_lat"].values if "event_lat" in df.columns else tpl_lat_arr)
        )
        ev_lon_arr = (
            df["lon"].values
            if "lon" in df.columns
            else (df["event_lon"].values if "event_lon" in df.columns else tpl_lon_arr)
        )

        df["det_amp_corr"] = self._distance_correct_amplitude_vec(
            df["event_amplitude"].values if "event_amplitude" in df.columns else np.full(len(df), np.nan),
            ev_lat_arr, ev_lon_arr, gamma,
        )

        # --- magnitude via amplitude ratio
        valid = (
            (df["tpl_amp_corr"] > 0)
            & (df["det_amp_corr"] > 0)
            & df["tpl_magnitude"].notna()
        )

        if valid.sum() == 0:
            raise ValueError("No valid amplitudes for magnitude estimation.")

        # Debug amplitude information
        if valid.sum() > 0:
            console.print(f"[dim]MAGNITUDE DEBUG INFO:[/dim]")
            console.print(f"[dim]  Valid detections: {valid.sum()}/{len(df)}[/dim]")
            console.print(f"[dim]  Template magnitude range: {df.loc[valid, 'tpl_magnitude'].min():.2f} to {df.loc[valid, 'tpl_magnitude'].max():.2f}[/dim]")
            console.print(f"[dim]  Template amplitude range: {df.loc[valid, 'tpl_amplitude'].min():.2e} to {df.loc[valid, 'tpl_amplitude'].max():.2e}[/dim]" if 'tpl_amplitude' in df.columns else "[dim]  Template amplitude: not available[/dim]")
            console.print(f"[dim]  Template amp corrected range: {df.loc[valid, 'tpl_amp_corr'].min():.2e} to {df.loc[valid, 'tpl_amp_corr'].max():.2e}[/dim]")
            console.print(f"[dim]  Detection amplitude range: {df.loc[valid, 'event_amplitude'].min():.2e} to {df.loc[valid, 'event_amplitude'].max():.2e}[/dim]" if 'event_amplitude' in df.columns else "[dim]  Detection amplitude: not available[/dim]")
            console.print(f"[dim]  Detection amp corrected range: {df.loc[valid, 'det_amp_corr'].min():.2e} to {df.loc[valid, 'det_amp_corr'].max():.2e}[/dim]")
            
            amp_ratios = df.loc[valid, "det_amp_corr"] / df.loc[valid, "tpl_amp_corr"]
            console.print(f"[dim]  Amplitude ratio range: {amp_ratios.min():.2e} to {amp_ratios.max():.2e}[/dim]")
            console.print(f"[dim]  log10(ratio) range: {np.log10(amp_ratios).min():.2f} to {np.log10(amp_ratios).max():.2f}[/dim]")

        df.loc[valid, "est_magnitude"] = (
            df.loc[valid, "tpl_magnitude"]
            + np.log10(df.loc[valid, "det_amp_corr"]
                       / df.loc[valid, "tpl_amp_corr"])
        )
        
        # Final magnitude debug info
        if valid.sum() > 0:
            final_mags = df.loc[valid, "est_magnitude"]
            console.print(f"[dim]  Final magnitude range: {final_mags.min():.2f} to {final_mags.max():.2f}[/dim]")
            console.print(f"[dim]  Final magnitude mean: {final_mags.mean():.2f} ± {final_mags.std():.2f}[/dim]")

        # --- uncertainty proxy (optional but useful)
        if "event_id" in df.columns:
            df["mag_std"] = df.groupby("event_id")["est_magnitude"].transform("std")

        console.print(
            f"🧮 Magnitudes estimated using amplitude ratios "
            f"({valid.sum()} valid detections)"
        )

        return df

    def estimate_magnitudes_all_methods(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate magnitudes using multiple amplitude representations (ratio method).

        Produces columns:
        - est_magnitude_max, est_magnitude_rms, est_magnitude_percentile, est_magnitude_envelope
        """
        df = detections_df.copy()

        # Only merge template info if the needed tpl_ columns are not already present
        need_tpl_cols = [
            "tpl_amplitude", "tpl_amplitude_max", "tpl_amplitude_rms",
            "tpl_amplitude_percentile", "tpl_amplitude_envelope",
            "tpl_magnitude", "tpl_lat", "tpl_lon",
        ]
        have_all = all(c in df.columns for c in need_tpl_cols if c.replace("tpl_", "") in template_info_df.columns)
        if not have_all:
            # Load mapping and determine merge key
            df, merge_on = self._load_mapping_and_merge(df)

            tpl_cols = [
                "amplitude", "amplitude_max", "amplitude_rms",
                "amplitude_percentile", "amplitude_envelope",
                "magnitude", "lat", "lon",
            ]
            # Select only template columns that exist and prefix them before merge
            tpl_subset = template_info_df[["template_id"] + [c for c in tpl_cols if c in template_info_df.columns]].copy()
            tpl_subset = tpl_subset.rename(columns={c: f"tpl_{c}" for c in tpl_cols if c in tpl_subset.columns})

            # Drop any tpl_ columns already in df to avoid suffix conflicts
            overlap = [c for c in tpl_subset.columns if c in df.columns and c != "template_id"]
            if overlap:
                df = df.drop(columns=overlap)

            n_before = len(df)
            df = df.merge(
                tpl_subset,
                left_on=merge_on,
                right_on="template_id",
                how="left",
                suffixes=("", "_dup")
            )
            # Drop any duplicate key column from the merge
            for c in list(df.columns):
                if c.endswith("_dup"):
                    df = df.drop(columns=[c])
            # If merge expanded rows (shouldn't with left join + unique keys),
            # deduplicate back to original size
            if len(df) > n_before:
                df = df.drop_duplicates(subset=df.columns.difference(
                    [c for c in df.columns if c.startswith("tpl_")]
                ).tolist()[:5], keep="first").head(n_before)

        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)

        # Vectorized distance corrections: compute once for tpl coords
        tpl_lat_arr = df["tpl_lat"].values if "tpl_lat" in df.columns else np.full(len(df), np.nan)
        tpl_lon_arr = df["tpl_lon"].values if "tpl_lon" in df.columns else np.full(len(df), np.nan)
        # Event coordinates for detections (prefer `lat`/`lon`, fall back to `event_lat`/`event_lon` or tpl coords)
        ev_lat_arr = (
            df["lat"].values
            if "lat" in df.columns
            else (df["event_lat"].values if "event_lat" in df.columns else tpl_lat_arr)
        )
        ev_lon_arr = (
            df["lon"].values
            if "lon" in df.columns
            else (df["event_lon"].values if "event_lon" in df.columns else tpl_lon_arr)
        )

        # For each amplitude method, compute corrected amplitudes and ratio magnitude
        methods = [
            ("", "event_amplitude", "tpl_amplitude"),
            ("_max", "event_amplitude_max", "tpl_amplitude_max"),
            ("_rms", "event_amplitude_rms", "tpl_amplitude_rms"),
            ("_percentile", "event_amplitude_percentile", "tpl_amplitude_percentile"),
            ("_envelope", "event_amplitude_envelope", "tpl_amplitude_envelope"),
        ]

        for suffix, det_col, tpl_col in methods:
            if det_col in df.columns and tpl_col in df.columns:
                df[f"tpl_amp_corr{suffix}"] = self._distance_correct_amplitude_vec(
                    df[tpl_col].values, tpl_lat_arr, tpl_lon_arr, gamma)
                df[f"det_amp_corr{suffix}"] = self._distance_correct_amplitude_vec(
                    df[det_col].values, ev_lat_arr, ev_lon_arr, gamma)
                valid = (
                    (df[f"tpl_amp_corr{suffix}"] > 0)
                    & (df[f"det_amp_corr{suffix}"] > 0)
                    & df.get("tpl_magnitude").notna()
                ) if "tpl_magnitude" in df.columns else (
                    (df[f"tpl_amp_corr{suffix}"] > 0)
                    & (df[f"det_amp_corr{suffix}"] > 0)
                )
                if valid.sum() > 0:
                    if "tpl_magnitude" in df.columns:
                        df.loc[valid, f"est_magnitude{suffix}"] = (
                            df.loc[valid, "tpl_magnitude"]
                            + np.log10(df.loc[valid, f"det_amp_corr{suffix}"] / df.loc[valid, f"tpl_amp_corr{suffix}"])
                        )
                    else:
                        df.loc[valid, f"est_magnitude{suffix}"] = np.log10(
                            df.loc[valid, f"det_amp_corr{suffix}"] / df.loc[valid, f"tpl_amp_corr{suffix}"]
                        )

        return df

    # ---------------------------------------------------------------------
    # Spectral Magnitude Estimation
    # ---------------------------------------------------------------------
    def estimate_magnitude_spectral(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Estimate magnitude from spectral fitting.

        Uses the low-frequency spectral level ratio between detected event
        and template to derive magnitude difference:

            M_det = M_tpl + (2/3) * log10(Omega_det / Omega_tpl)

        where Omega is the low-frequency plateau of the displacement spectrum.
        This is more robust to path and site effects than time-domain amplitude.
        """
        df = detections_df.copy()
        df, merge_on = self._load_mapping_and_merge(df)

        tpl_cols = ["amplitude", "magnitude", "lat", "lon"]
        df = df.merge(
            template_info_df[["template_id"] + [c for c in tpl_cols if c in template_info_df.columns]],
            left_on=merge_on, right_on="template_id", how="left", suffixes=("", "_tpl")
        )
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])
        for c in tpl_cols:
            if c in df.columns:
                df.rename(columns={c: f"tpl_{c}"}, inplace=True)

        # Compute spectral ratio proxy using existing amplitude columns
        # True spectral fitting requires raw waveform access; here we approximate
        # using the envelope amplitude as a proxy for Omega_0.
        det_col = 'event_amplitude_envelope'
        tpl_col = 'tpl_amplitude_envelope' if 'tpl_amplitude_envelope' in df.columns else 'tpl_amplitude'

        if det_col not in df.columns:
            det_col = 'event_amplitude'

        valid = (
            df[det_col].notna() & (df[det_col] > 0) &
            df[tpl_col].notna() & (df[tpl_col] > 0) &
            df.get('tpl_magnitude', pd.Series(dtype=float)).notna()
        )

        if valid.sum() > 0:
            df.loc[valid, 'est_magnitude_spectral'] = (
                df.loc[valid, 'tpl_magnitude']
                + (2.0 / 3.0) * np.log10(df.loc[valid, det_col] / df.loc[valid, tpl_col])
            )

        console.print(f"🔬 Spectral magnitudes estimated ({valid.sum()} valid)")
        return df

    # ---------------------------------------------------------------------
    # Station Corrections
    # ---------------------------------------------------------------------
    def compute_station_corrections(
        self,
        detections_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute per-template station magnitude corrections.

        If templates have known magnitudes and we have multiple detections
        per template, the median residual (est - known) gives a station/path
        correction term.

        Returns dict mapping template_id → correction (to subtract from est_magnitude).
        """
        df = detections_df.copy()
        if 'tpl_magnitude' not in df.columns or 'est_magnitude' not in df.columns:
            return {}

        valid = df['tpl_magnitude'].notna() & df['est_magnitude'].notna()
        if valid.sum() < 5:
            return {}

        residuals = df.loc[valid].groupby('template_id').apply(
            lambda g: (g['est_magnitude'] - g['tpl_magnitude']).median()
        )
        corrections = residuals.to_dict()
        console.print(f"📐 Station corrections computed for {len(corrections)} templates "
                       f"(median correction: {np.median(list(corrections.values())):.3f})")
        return corrections

    def apply_station_corrections(
        self,
        df: pd.DataFrame,
        corrections: Dict,
    ) -> pd.DataFrame:
        """Apply station corrections to estimated magnitudes."""
        df = df.copy()
        if corrections and 'est_magnitude' in df.columns:
            df['station_correction'] = df['template_id'].map(corrections).fillna(0.0)
            df['est_magnitude_corrected'] = df['est_magnitude'] - df['station_correction']
        return df

    # ---------------------------------------------------------------------
    # Magnitude Uncertainty
    # ---------------------------------------------------------------------
    def compute_magnitude_uncertainty(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add magnitude uncertainty columns.

        - mag_uncertainty_ratio: from propagation of log-amplitude ratio errors
        - mag_uncertainty_bootstrap: bootstrap resampling of per-template estimates
        """
        df = df.copy()

        # Simple uncertainty from similarity (lower CC → higher uncertainty)
        if 'similarity' in df.columns and 'est_magnitude' in df.columns:
            # Empirical: sigma ≈ 0.4 / similarity (rough calibration)
            df['mag_uncertainty'] = 0.4 / df['similarity'].clip(lower=0.1)

        # Per-template spread
        if 'est_magnitude' in df.columns and 'template_id' in df.columns:
            df['mag_uncertainty_template'] = df.groupby('template_id')['est_magnitude'].transform('std')

        return df

    def _amplitude_diagnostic_checks(self, df: pd.DataFrame, label: str = "") -> None:
        """Run quick diagnostic checks on amplitudes/magnitudes and print summaries.

        Intended to be run after magnitude estimation (when tpl_amp_corr and
        det_amp_corr are available) to quickly spot issues described by the
        vertical-strip artifact (identical distance correction for det/tpl).
        """
        if df is None or len(df) == 0:
            console.print("[dim]Amplitude diagnostics: no data available[/dim]")
            return

        console.print(f"[cyan]🔎 Amplitude diagnostics {label}[/cyan]")

        def _safe_nunique(col):
            return int(df[col].nunique()) if col in df.columns else None

        def _safe_minmax(col):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) > 0:
                    return float(s.min()), float(s.max()), float(s.median())
            return None

        utpl = _safe_nunique("template_id")
        console.print(f"- Unique templates: {utpl if utpl is not None else 'n/a'}")

        if "tpl_magnitude" in df.columns:
            console.print(f"- Template magnitude unique: {df['tpl_magnitude'].nunique()}  range: {df['tpl_magnitude'].min():.2f} to {df['tpl_magnitude'].max():.2f}")
        elif "magnitude" in df.columns:
            console.print(f"- Template magnitude column 'magnitude' present: unique {df['magnitude'].nunique()}  range: {df['magnitude'].min():.2f} to {df['magnitude'].max():.2f}")
        else:
            console.print("- Template magnitude: not available")

        for col in ("event_amplitude", "tpl_amplitude", "tpl_amp_corr", "det_amp_corr"):
            mm = _safe_minmax(col)
            if mm is not None:
                console.print(f"- {col}: min={mm[0]:.2e} max={mm[1]:.2e} median={mm[2]:.2e}")
            else:
                console.print(f"- {col}: not available")

        # Ratio stats
        if "tpl_amp_corr" in df.columns and "det_amp_corr" in df.columns:
            mask = (pd.to_numeric(df["tpl_amp_corr"], errors="coerce") > 0) & (pd.to_numeric(df["det_amp_corr"], errors="coerce") > 0)
            n_mask = int(mask.sum())
            if n_mask > 0:
                ratio = (df.loc[mask, "det_amp_corr"].astype(float) / df.loc[mask, "tpl_amp_corr"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
                if len(ratio) > 0:
                    lr = np.log10(ratio)
                    console.print(f"- Valid ratio pairs: {len(ratio)}  log10(ratio): min={lr.min():.2f} max={lr.max():.2f} median={np.median(lr):.2f}")
                else:
                    console.print(f"- Valid ratio pairs: {n_mask} but ratio values invalid after cleaning")
            else:
                console.print("- No valid tpl_amp_corr>0 & det_amp_corr>0 pairs for ratio diagnostics")

        console.print("[cyan]End diagnostics[/cyan]\n")
        
    # ---------------------------------------------------------------------
    # Bayesian Magnitude Estimation
    # ---------------------------------------------------------------------

    def estimate_magnitudes_bayes(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame,
        n_samples: int = 2000,
        n_tune: int = 1000
    ) -> Tuple[pd.DataFrame, object]:
        """
        Improved Bayesian magnitude estimation using template amplitudes as calibration.
        
        Model: M = α * log10(A) + β + f(distance) + ε
        
        Where:
        - α, β are learned from template data
        - f(distance) accounts for geometric spreading
        - ε is measurement error
        """
        
        df = detections_df.copy()
        
        # Load mapping and determine merge key
        df, merge_on = self._load_mapping_and_merge(df)
        
        # Merge mit template info - Join auf rechter 'template_id' ohne Konflikte
        df = df.merge(
            template_info_df[["template_id", "amplitude", "magnitude", "lat", "lon"]],
            left_on=merge_on,
            right_on="template_id",
            how="left",
            suffixes=("", "_tpl")
        )
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])
        
        # Distance correction parameters
        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)
        R0 = self.config.get("evaluation.reference_distance", 100.0)
        
        # Calculate distances separately for templates and detections
        tpl_lat_arr = (
            df["lat_tpl"].values
            if "lat_tpl" in df.columns
            else (df["tpl_lat"].values if "tpl_lat" in df.columns else np.full(len(df), np.nan))
        )
        tpl_lon_arr = (
            df["lon_tpl"].values
            if "lon_tpl" in df.columns
            else (df["tpl_lon"].values if "tpl_lon" in df.columns else np.full(len(df), np.nan))
        )

        ev_lat_arr = (
            df["lat"].values
            if "lat" in df.columns
            else (df["event_lat"].values if "event_lat" in df.columns else np.full(len(df), np.nan))
        )
        ev_lon_arr = (
            df["lon"].values
            if "lon" in df.columns
            else (df["event_lon"].values if "event_lon" in df.columns else np.full(len(df), np.nan))
        )

        # Distance arrays (km)
        tpl_dist_km = self._compute_distances_vec(tpl_lat_arr, tpl_lon_arr)
        det_dist_km = self._compute_distances_vec(ev_lat_arr, ev_lon_arr)

        # Determine amplitude column names after the merge (template amplitudes usually suffixed)
        tpl_amp_col = "amplitude_tpl" if "amplitude_tpl" in df.columns else ("amplitude" if "amplitude" in df.columns else None)

        # Apply distance correction to amplitudes using correct coords
        df["tpl_amp_corr"] = self._distance_correct_amplitude_vec(
            df[tpl_amp_col].values if tpl_amp_col is not None else np.full(len(df), np.nan),
            tpl_lat_arr, tpl_lon_arr, gamma,
        )
        df["det_amp_corr"] = self._distance_correct_amplitude_vec(
            df["event_amplitude"].values if "event_amplitude" in df.columns else np.full(len(df), np.nan),
            ev_lat_arr, ev_lon_arr, gamma,
        )
        
        # Filter valid data
        # Template magnitude after merge may be in 'magnitude_tpl'
        tpl_mag_col = "magnitude_tpl" if "magnitude_tpl" in df.columns else ("magnitude" if "magnitude" in df.columns else None)

        valid = (
            (df["tpl_amp_corr"] > 0) & 
            (df["det_amp_corr"] > 0) & 
            (df[tpl_mag_col].notna() if tpl_mag_col is not None else False)
        )
        
        if valid.sum() == 0:
            raise ValueError("No valid amplitudes for Bayesian magnitude estimation!")
        
        # Prepare data for PyMC model
        log_tpl_amp = np.log10(df.loc[valid, "tpl_amp_corr"].values)
        log_det_amp = np.log10(df.loc[valid, "det_amp_corr"].values)
        tpl_mag = df.loc[valid, tpl_mag_col].values
        
        console.print(f"📊 Building Bayesian model with {valid.sum()} data points...")

        # Lazy import PyMC to avoid heavy dependency at module import time
        try:
            import pymc as pm
        except Exception as e:
            raise RuntimeError(f"Bayesian estimation requires 'pymc' but import failed: {e}")

        with pm.Model() as model:
            # Priors for linear relationship: M = α * log10(A) + β
            α = pm.Normal("α", mu=1.0, sigma=0.5)  # Should be ~1 for amplitude-magnitude scaling
            β = pm.Normal("β", mu=0.0, sigma=2.0)  # Station correction term
            
            # Measurement error
            σ = pm.HalfNormal("σ", sigma=0.5)
            
            # Template magnitudes (observed)
            μ_tpl = α * log_tpl_amp + β
            M_tpl_obs = pm.Normal("M_tpl_obs", mu=μ_tpl, sigma=σ, observed=tpl_mag)
            
            # Trace
            trace = pm.sample(
                draws=n_samples,
                tune=n_tune,
                chains=2,
                target_accept=0.95,
                cores=2,
                progressbar=True
            )
        
        # Extract posterior summaries
        α_post = trace.posterior["α"].mean().item()
        β_post = trace.posterior["β"].mean().item()
        σ_post = trace.posterior["σ"].mean().item()
        
        # Estimate magnitudes for detections
        df.loc[valid, "est_magnitude"] = α_post * log_det_amp + β_post
        df.loc[valid, "est_magnitude_std"] = σ_post
        
        # Also compute magnitude using amplitude ratio method for comparison
        df.loc[valid, "ratio_magnitude"] = (
            df.loc[valid, "magnitude"] + 
            np.log10(df.loc[valid, "det_amp_corr"] / df.loc[valid, "tpl_amp_corr"])
        )
        
        console.print("✅ Bayesian magnitude estimation complete!")
        console.print(f"   α = {α_post:.3f} ± {trace.posterior['α'].std().item():.3f}")
        console.print(f"   β = {β_post:.3f} ± {trace.posterior['β'].std().item():.3f}")
        console.print(f"   σ = {σ_post:.3f}")
        
        # Generate comprehensive QC plots
        self._plot_bayesian_qc(df, trace, log_tpl_amp, log_det_amp, tpl_mag)
        
        return df, trace

    # ---------------------------------------------------------------------
    # Robust, Nonlinear Magnitude Calibration (Monotonic Mapping)
    # ---------------------------------------------------------------------
    def estimate_magnitudes_robust(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame,
        method: str = "isotonic"
    ) -> pd.DataFrame:
        """
        Robust magnitude estimation using template amplitude–magnitude calibration.

        Approach:
        1. Fit a robust linear regression M = slope * log10(A_corr) + intercept
           from template data (Huber regressor, resistant to outliers).
        2. Optionally refine with isotonic regression *inside* the training range
           to capture non-linearities, but always use linear extrapolation
           outside the training amplitude range so that large/small events
           are not clipped.
        3. The linear model is the physical expectation: M ∝ log10(A).
        """

        # Prepare template calibration data
        tpl = template_info_df.copy()
        tpl = tpl.rename(columns={"amplitude": "tpl_amplitude", "magnitude": "tpl_magnitude"})

        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)
        R0 = self.config.get("evaluation.reference_distance", 1.0)
        st_lat = self.config["stations.lat"]
        st_lon = self.config["stations.lon"]

        # Distance correction for templates
        tpl["tpl_amp_corr"] = self._distance_correct_amplitude_vec(
            tpl["tpl_amplitude"].values if "tpl_amplitude" in tpl.columns else np.full(len(tpl), np.nan),
            tpl["lat"].values if "lat" in tpl.columns else np.full(len(tpl), np.nan),
            tpl["lon"].values if "lon" in tpl.columns else np.full(len(tpl), np.nan),
            gamma,
        )
        valid_tpl = tpl[tpl["tpl_amp_corr"].notna() & tpl["tpl_magnitude"].notna()]
        if len(valid_tpl) < 3:
            raise ValueError("Not enough valid templates for magnitude calibration.")

        x_tpl = np.log10(valid_tpl["tpl_amp_corr"].values)
        y_tpl = valid_tpl["tpl_magnitude"].values

        # Always fit a robust linear model (the physical baseline)
        huber = HuberRegressor()
        huber.fit(x_tpl.reshape(-1, 1), y_tpl)
        slope_lin = float(huber.coef_[0])
        intercept_lin = float(huber.intercept_)

        # Optionally fit isotonic for non-linear refinement inside the range
        iso = None
        x_min, x_max = float(x_tpl.min()), float(x_tpl.max())
        if method == "isotonic" and len(valid_tpl) >= 5:
            try:
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso.fit(x_tpl, y_tpl)
            except Exception:
                iso = None

        def predict_magnitude(x_arr: np.ndarray) -> np.ndarray:
            """Predict magnitude: isotonic inside training range, linear outside."""
            y = np.empty_like(x_arr)
            if iso is not None:
                # Inside training range: use isotonic
                inside = (x_arr >= x_min) & (x_arr <= x_max)
                outside_lo = x_arr < x_min
                outside_hi = x_arr > x_max
                if inside.any():
                    y[inside] = iso.predict(x_arr[inside])
                # Outside: linear extrapolation from the robust fit
                if outside_lo.any():
                    y[outside_lo] = slope_lin * x_arr[outside_lo] + intercept_lin
                if outside_hi.any():
                    y[outside_hi] = slope_lin * x_arr[outside_hi] + intercept_lin
            else:
                # Pure linear
                y = slope_lin * x_arr + intercept_lin
            return y

        # Prepare detections with corrected amplitudes
        df = detections_df.copy()
        df, merge_on = self._load_mapping_and_merge(df)

        df = df.merge(
            template_info_df[["template_id", "lat", "lon"]],
            left_on=merge_on,
            right_on="template_id",
            how="left",
            suffixes=("", "_tpl")
        )
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])

        ev_lat_arr = (
            df["lat"].values
            if "lat" in df.columns
            else (df["event_lat"].values if "event_lat" in df.columns else np.full(len(df), np.nan))
        )
        ev_lon_arr = (
            df["lon"].values
            if "lon" in df.columns
            else (df["event_lon"].values if "event_lon" in df.columns else np.full(len(df), np.nan))
        )

        df["det_amp_corr"] = self._distance_correct_amplitude_vec(
            df["event_amplitude"].values if "event_amplitude" in df.columns else np.full(len(df), np.nan),
            ev_lat_arr, ev_lon_arr, gamma,
        )
        valid_det = df[df["det_amp_corr"].notna() & (df["det_amp_corr"] > 0)]
        if len(valid_det) == 0:
            raise ValueError("No valid detections for robust magnitude estimation.")

        x_det = np.log10(valid_det["det_amp_corr"].values)
        y_pred = predict_magnitude(x_det)

        df.loc[valid_det.index, "est_magnitude"] = y_pred

        method_label = "isotonic + linear extrapolation" if iso is not None else "huber linear"
        console.print(
            f"✅ Robust magnitude estimation completed ({method_label}).\n"
            f"   Calibration: M = {slope_lin:.3f} * log10(A) + {intercept_lin:.3f}  "
            f"(from {len(valid_tpl)} templates, amp range 10^{x_min:.1f}–10^{x_max:.1f})"
        )

        return df

    def _plot_bayesian_qc(self, df, trace, log_tpl_amp, log_det_amp, tpl_mag):
        """Generate comprehensive QC plots for Bayesian magnitude estimation."""
        plots_dir = self.config.get_path("plots_dir")
        plots_dir.mkdir(exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        # Extract posterior samples
        α_samples = trace.posterior["α"].values.flatten()
        β_samples = trace.posterior["β"].values.flatten()
        
        # Plot 1: Template calibration
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.scatter(log_tpl_amp, tpl_mag, s=10, alpha=0.5, label="Templates")
        
        # Plot posterior samples
        x_range = np.linspace(log_tpl_amp.min()-0.5, log_tpl_amp.max()+0.5, 100)
        for i in np.random.randint(0, len(α_samples), 50):
            plt.plot(x_range, α_samples[i]*x_range + β_samples[i], 
                    'r-', alpha=0.1, lw=0.5)
        
        plt.plot(x_range, np.mean(α_samples)*x_range + np.mean(β_samples), 
                'k-', lw=2, label="Posterior mean")
        plt.xlabel("log10(Corrected Template Amplitude)")
        plt.ylabel("Template Magnitude")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("Template Calibration")
        
        # Plot 2: Detection magnitudes comparison
        plt.subplot(132)
        valid = df["est_magnitude"].notna()
        plt.scatter(df.loc[valid, "ratio_magnitude"], 
                    df.loc[valid, "est_magnitude"], 
                    s=8, alpha=0.5)
        plt.plot([-2, 4], [-2, 4], 'k--', alpha=0.5, label="1:1 line")
        plt.xlabel("Ratio Method Magnitude")
        plt.ylabel("Bayesian Magnitude")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.title("Method Comparison")
        
        # Plot 3: Magnitude residuals
        plt.subplot(133)
        plt.hist(df.loc[valid, "ratio_magnitude"] - df.loc[valid, "est_magnitude"], 
                bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Ratio - Bayesian Magnitude")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.title(f"Residuals (mean={np.mean(df.loc[valid, 'ratio_magnitude'] - df.loc[valid, 'est_magnitude']):.3f})")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "qc_bayesian_magnitude_comprehensive.png", dpi=200)
        plt.close()
        
        # Plot 4: Posterior distributions
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        # Lazy import ArviZ only when plotting Bayes posteriors
        try:
            import arviz as az
        except Exception:
            console.print("[yellow]ArviZ not installed; skipping posterior plots[/yellow]")
            return

        az.plot_posterior(trace, var_names=["α"], ax=axes[0])
        axes[0].set_title("Posterior: α")
        
        az.plot_posterior(trace, var_names=["β"], ax=axes[1])
        axes[1].set_title("Posterior: β")
        
        az.plot_posterior(trace, var_names=["σ"], ax=axes[2])
        axes[2].set_title("Posterior: σ")
        
        plt.tight_layout()
        plt.savefig(plots_dir / "qc_bayesian_posteriors.png", dpi=200)
        plt.close()

    def write_pyrocko_marker_file(self, detections_df: pd.DataFrame, output_file: Path):
        def fmt_time(t): return t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        station_lat = self.config.get("stations.lat", np.nan)
        station_lon = self.config.get("stations.lon", np.nan)

        lines = []
        ev_id = 1
        for _, row in detections_df.iterrows():
            mag = row.get("est_magnitude")
            if pd.isna(mag):
                continue

            name = f"ev_{ev_id}"
            if "cluster_id" in row.index and not pd.isna(row.get("cluster_id", np.nan)):
                name += f" (cluster {int(row['cluster_id'])})"
            if "n_templates" in row.index and not pd.isna(row.get("n_templates", np.nan)):
                name += f" n={int(row['n_templates'])}"

            time = row.get("time")
            if pd.isna(time):
                continue

            # Support both 'lat'/'lon' and 'tpl_lat'/'tpl_lon' column names
            lat = row.get("lat", row.get("tpl_lat", station_lat))
            lon = row.get("lon", row.get("tpl_lon", station_lon))
            if pd.isna(lat):
                lat = station_lat
            if pd.isna(lon):
                lon = station_lon

            lines.extend([
                f"name = {name}",
                f"time = {fmt_time(pd.to_datetime(time))}",
                f"latitude = {lat:.5f}",
                f"longitude = {lon:.5f}",
                f"magnitude = {mag:.4f}",
                "catalog = template_matching",
                "-"*44
            ])
            ev_id += 1

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("\n".join(lines))
        console.print(f"🗺️ Pyrocko marker file written: [cyan]{output_file}[/cyan] ({ev_id-1} events)")

    def _map_template_ids_for_plotting(self, detections_df: pd.DataFrame, 
                                    template_info_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare template info DataFrame with correct IDs for plotting."""
        
        # Versuche Mapping-Datei zu laden
        mapping_file = self.config.get_path("base_dir") / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        if mapping_file.exists():
            mapping_df = pd.read_csv(mapping_file)
            
            # Merge template info with mapping
            tpl_mapped = template_info_df.merge(
                mapping_df[["original_template_id", "representative_id"]].rename(
                    columns={"original_template_id": "template_id"}
                ),
                on="template_id",
                how="left"
            )
            
            # Filter: nur Templates, die tatsächlich verwendet wurden (in detections)
            used_repr_ids = detections_df["template_id"].dropna().unique()
            tpl_mapped = tpl_mapped[tpl_mapped["representative_id"].isin(used_repr_ids)]
            
            # Aktualisiere template_id für konsistente Verwendung
            tpl_mapped["template_id"] = tpl_mapped["representative_id"]
            
            return tpl_mapped
        else:
            # Fallback: verwende originale IDs
            console.print("[yellow]⚠️ No template mapping found. Plot may show incorrect templates.[/yellow]")
            return template_info_df
        
    def simple_deduplicate_detections(self, df: pd.DataFrame, time_window_sec: float = 1.0) -> pd.DataFrame:
        """
        Simple deduplication based on time proximity.
        """
        if len(df) == 0:
            return df
        
        # Sort by time
        df_sorted = df.sort_values('time').reset_index(drop=True)

        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted['time']):
            df_sorted['time'] = pd.to_datetime(df_sorted['time'])
        
        # Vectorized event grouping
        times = df_sorted['time'].values
        diffs = np.diff(times).astype('timedelta64[ms]').astype(np.float64) / 1000.0
        event_ids = np.concatenate([[0], np.cumsum(diffs > time_window_sec)]).astype(np.int64)
        
        df_sorted['event_id'] = event_ids
        
        # Keep only the best detection per event (highest similarity)
        deduplicated = df_sorted.loc[df_sorted.groupby('event_id')['similarity'].idxmax()]
        
        return deduplicated.reset_index(drop=True)

    def build_final_catalog(
        self,
        df: pd.DataFrame,
        time_window_sec: float = 2.0,
    ) -> pd.DataFrame:
        """Build a final event catalog where each physical event appears once.

        Within each time window, all detections (from different templates) are
        grouped into one event.  The output keeps:
        - *time*: time of the best-CC detection (representative pick)
        - *similarity*: maximum CC across templates
        - *n_templates*: number of templates that detected this event
        - *est_magnitude*: similarity-weighted mean of individual estimates
        - *est_magnitude_std*: standard deviation across template estimates
        - *est_magnitude_min / _max*: range
        - *event_amplitude*: amplitude of the best-CC detection
        - location columns from the best-CC detection

        Parameters
        ----------
        df : DataFrame with all (possibly duplicated) detections including
             ``est_magnitude``, ``similarity``, ``time``, etc.
        time_window_sec : seconds within which detections are the same event.

        Returns
        -------
        Deduplicated DataFrame, one row per physical event.
        """
        if len(df) == 0:
            return df

        df = df.copy()
        # Ensure time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        df_sorted = df.sort_values('time').reset_index(drop=True)

        # --- Assign event group IDs (vectorized gap detection) ---
        times = df_sorted['time'].values  # numpy datetime64
        diffs = np.diff(times).astype('timedelta64[ms]').astype(np.float64) / 1000.0
        gaps = diffs > time_window_sec
        event_ids = np.concatenate([[0], np.cumsum(gaps)]).astype(np.int64)
        df_sorted['_event_group'] = event_ids

        # --- Aggregate each group ---
        rows = []
        for gid, grp in df_sorted.groupby('_event_group', sort=True):
            best_idx = grp['similarity'].idxmax()
            best = grp.loc[best_idx]

            mags = grp['est_magnitude'].dropna()
            sims = grp.loc[mags.index, 'similarity'] if len(mags) > 0 else pd.Series(dtype=float)

            if len(mags) > 0 and sims.sum() > 0:
                # Similarity-weighted mean magnitude
                weights = sims.values
                weights = weights / weights.sum()
                mag_mean = float(np.average(mags.values, weights=weights))
                mag_std = float(np.sqrt(np.average((mags.values - mag_mean) ** 2, weights=weights)))
            elif len(mags) > 0:
                mag_mean = float(mags.mean())
                mag_std = float(mags.std())
            else:
                mag_mean = np.nan
                mag_std = np.nan

            row = {
                'time': best['time'],
                'similarity': float(best['similarity']),
                'n_templates': len(grp),
                'template_id': int(best['template_id']) if pd.notna(best.get('template_id')) else -1,
                'event_amplitude': float(best.get('event_amplitude', np.nan)),
                'detection_snr': float(best.get('detection_snr', np.nan)),
                'cc_sharpness': float(best.get('cc_sharpness', np.nan)),
                'n_stations': int(best.get('n_stations', 1)) if pd.notna(best.get('n_stations')) else 1,
                'est_magnitude': mag_mean,
                'est_magnitude_std': mag_std,
                'est_magnitude_min': float(mags.min()) if len(mags) > 0 else np.nan,
                'est_magnitude_max': float(mags.max()) if len(mags) > 0 else np.nan,
                'event_id': int(gid),
            }

            # Carry over location columns from best detection
            for col in ('lat', 'lon', 'tpl_lat', 'tpl_lon', 'lat_tpl', 'lon_tpl', 'depth'):
                if col in best.index and pd.notna(best[col]):
                    row[col] = float(best[col])

            # Spectral / corrected magnitude: weighted mean if available
            for mcol in ('est_magnitude_spectral', 'est_magnitude_corrected'):
                vals = grp[mcol].dropna() if mcol in grp.columns else pd.Series(dtype=float)
                if len(vals) > 0:
                    w = grp.loc[vals.index, 'similarity'].values
                    w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
                    row[mcol] = float(np.average(vals.values, weights=w))

            rows.append(row)

        catalog = pd.DataFrame(rows)
        n_orig = len(df_sorted)
        console.print(
            f"📋 Final catalog: {len(catalog)} unique events "
            f"(from {n_orig} raw detections, "
            f"avg {n_orig / max(len(catalog), 1):.0f} templates/event)"
        )
        return catalog
    
    def plot_detection_and_template_waveforms(
        self,
        detections_df: pd.DataFrame,
        template_info_df: pd.DataFrame,
        output_dir: Path,
        n_samples: int = 50
    ):
        """
        Generate QC plots comparing detection waveforms to templates.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map template IDs for plotting
        tpl_for_plotting = self._map_template_ids_for_plotting(detections_df, template_info_df)
        
        # Sample detections for plotting
        sampled_detections = detections_df.sample(
            n=min(n_samples, len(detections_df)), random_state=42
        )
        
        plots_created = 0
        for _, det_row in sampled_detections.iterrows():
            try:
                tpl_id = det_row["template_id"]
                tpl_row = tpl_for_plotting[tpl_for_plotting["template_id"] == tpl_id]
                if tpl_row.empty:
                    continue
                # Compute template start and end times from event_name
                tpl_event_time = name2time(tpl_row.iloc[0]["event_name"])
                tpl_pre_event = self.config.get("template_creation.pre_event", 0.5)
                tpl_post_event = self.config.get("template_creation.post_event", 10.0)
                tpl_start_time = tpl_event_time - pd.Timedelta(seconds=tpl_pre_event)
                tpl_end_time = tpl_event_time + pd.Timedelta(seconds=tpl_post_event)
                
                template_trace = self.load_waveforms(
                    starttime=pd.to_datetime(tpl_start_time, utc=True),
                    endtime=pd.to_datetime(tpl_end_time, utc=True),
                    station=self.config["stations.station_code"],
                    channel=self.config.get("stations.primary_channel", "Z")
                )
                detected_trace = self.load_waveforms(
                    starttime=pd.to_datetime(det_row["time"], utc=True) - pd.Timedelta(seconds=self.config.get("template_matching.pre_event", 0.5)),
                    endtime=pd.to_datetime(det_row["time"], utc=True) + pd.Timedelta(seconds=self.config.get("template_matching.post_event", 5.0)),
                    station=self.config["stations.station_code"],
                    channel=self.config.get("stations.primary_channel", "Z")
                )
                
                if template_trace is not None and detected_trace is not None:
                    plot_waveform_comparison(
                        template_trace=template_trace,
                        detected_trace=detected_trace,
                        outdir=output_dir,
                        event_id=f"det_{int(det_row.name)}_tpl_{tpl_id}",
                        similarity=det_row.get("similarity", np.nan)
                    )
                    plots_created += 1
                else:
                    console.print(f"[dim]Warning: Could not load waveforms for detection {det_row.name}[/dim]")
                    
            except Exception as e:
                console.print(f"[dim]Warning: Failed to create plot for detection {det_row.name}: {e}[/dim]")
                continue
            
        if plots_created > 0:
            console.print(f"🖼️ {plots_created} waveform comparison plots saved to: [cyan]{output_dir}[/cyan]")
        else:
            console.print(f"[yellow]No waveform comparison plots created - check waveform data availability[/yellow]")

    def evaluate(self, detection_dir: Path, template_info_file: Path,
                progress: Optional[Progress] = None,
                min_similarity: float = 0.0,
                use_bayesian: bool = True) -> Dict:
        """
        Main evaluation workflow with option for Bayesian or ratio method.
        """
        
        console.print("[cyan]📊 Results Evaluation[/cyan]")
        
        # Load data
        det_files = sorted(detection_dir.glob("detections_*.csv"))
        if not det_files:
            return {"success": False}
        
        det = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in det_files], 
                        ignore_index=True)
        det = det[det["similarity"] >= min_similarity]

        # Post-hoc edge filter: remove detections near day boundaries
        # that are caused by filter ringing / CC normalization artifacts.
        edge_sec = float(self.config.get('detection_qc.edge_margin_seconds', 10.0))
        if edge_sec > 0:
            times = det['time']
            sod = times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second + times.dt.microsecond / 1e6
            edge_mask = (sod < edge_sec) | (sod > 86400 - edge_sec)
            n_edge = edge_mask.sum()
            if n_edge > 0:
                det = det[~edge_mask].reset_index(drop=True)
                console.print(f"[dim]Edge filter: removed {n_edge} detections within {edge_sec:.0f}s of day boundaries[/dim]")
        tpl = pd.read_csv(template_info_file)
        
        console.print(f"📈 Loaded {len(det)} detections from {len(tpl)} templates")
        
        # Choose estimation method (config override supported)
        cal_method = str(self.config.get("evaluation.calibration_method", "bayes")).lower()
        trace = None
        if cal_method in ("bayes", "bayesian"):
            console.print("[bold]Using Bayesian magnitude estimation[/bold]")
            try:
                det, trace = self.estimate_magnitudes_bayes(det, tpl)
                method = "bayesian"
            except Exception as e:
                console.print(f"[yellow]Bayesian estimation unavailable ({e}). Falling back to robust isotonic.[/yellow]")
                det = self.estimate_magnitudes_robust(det, tpl, method="isotonic")
                method = "robust"
        elif cal_method in ("robust", "isotonic"):
            console.print("[bold]Using robust monotonic calibration (isotonic)[/bold]")
            det = self.estimate_magnitudes_robust(det, tpl, method="isotonic")
            method = "robust"
        elif cal_method in ("ratio", "amp_ratio"):
            console.print("[bold]Using amplitude ratio magnitude estimation[/bold]")
            det = self.estimate_magnitudes(det, tpl)
            method = "ratio"
        else:
            console.print(f"[yellow]Unknown calibration method '{cal_method}', defaulting to ratio[/yellow]")
            det = self.estimate_magnitudes(det, tpl)
            method = "ratio"

        # Diagnostic checks on amplitudes/magnitudes before deduplication
        try:
            self._amplitude_diagnostic_checks(det, label=method)
        except Exception as e:
            console.print(f"[yellow]Amplitude diagnostics failed: {e}[/yellow]")

        det = self.simple_deduplicate_detections(det, time_window_sec=1.0)

        # --- Optional magnitude refinement steps (controlled by config) ---

        # Spectral magnitude estimation
        if self.config.get('evaluation.compute_spectral', False):
            try:
                det = self.estimate_magnitude_spectral(det, tpl)
            except Exception as e:
                console.print(f"[yellow]Spectral magnitude estimation failed: {e}[/yellow]")

        # Station corrections
        if self.config.get('evaluation.compute_station_corrections', True):
            try:
                corrections = self.compute_station_corrections(det)
                if corrections:
                    det = self.apply_station_corrections(det, corrections)
            except Exception as e:
                console.print(f"[yellow]Station corrections failed: {e}[/yellow]")

        # Magnitude uncertainties
        if self.config.get('evaluation.compute_uncertainty', True):
            try:
                det = self.compute_magnitude_uncertainty(det)
            except Exception as e:
                console.print(f"[yellow]Magnitude uncertainty computation failed: {e}[/yellow]")

        # Multi-amplitude-method magnitudes (separate CSV; slowest step)
        det_multi = None
        if self.config.get('evaluation.compute_all_amplitude_methods', False):
            det_multi = self.estimate_magnitudes_all_methods(det, tpl)

        # Build final single-event catalog: collapse remaining duplicates
        # (from the _load_mapping_and_merge expansion) into one row per event
        # with CC-weighted mean magnitudes.
        catalog = self.build_final_catalog(det, time_window_sec=2.0)
        
        # Save results
        output_dir = self.config.get_path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Primary output: deduplicated final catalog (one row per event)
        csv_file = output_dir / f"detections_with_magnitude.csv"
        catalog.to_csv(csv_file, index=False, float_format="%.5f")
        if det_multi is not None:
            csv_multi = output_dir / f"detections_with_magnitude_multi.csv"
            det_multi.to_csv(csv_multi, index=False, float_format="%.5f")
        else:
            csv_multi = None
        
        # Create marker file from final catalog
        self.write_pyrocko_marker_file(catalog, output_dir / f"detections.markers")

        
        
        # Apply minimum magnitude filter for b-value calculation and plotting
        min_mag = self.config.get('evaluation.min_magnitude', 0.0)
        catalog_full = catalog.copy()  # Keep full catalog for output files
        if min_mag != 0.0:  # Apply filter for any non-zero threshold (positive or negative)
            catalog_filtered = catalog[catalog["est_magnitude"] >= min_mag].copy()
            n_filtered = len(catalog) - len(catalog_filtered)
            if n_filtered > 0:
                console.print(f"[dim]Filtered {n_filtered} events below magnitude {min_mag:.1f} for analysis[/dim]")
        else:
            catalog_filtered = catalog.copy()
        
        # Calculate b-value on the filtered catalog
        if len(catalog_filtered) > 0:
            bval = self.calc_bvalue(catalog_filtered["est_magnitude"].values)
        else:
            console.print("[yellow]Warning: No events above min_magnitude for b-value calculation[/yellow]")
            bval = {"b": np.nan, "Mc": np.nan, "a": np.nan, "n_events": 0}
        
        # Generate plots
        plots_dir = self.config.get_path("plots_dir")
        plots_dir.mkdir(parents=True, exist_ok=True)
        # Attempt to produce catalog comparison plots if original catalog is available
        try:
            orig_catalog = None
            # Try config.paths.catalog_file first, then fallback to get_path
            try:
                catalog_path = None
                if hasattr(self.config, "paths") and hasattr(self.config.paths, "catalog_file"):
                    catalog_path = Path(self.config.paths.catalog_file)
                else:
                    try:
                        cp = self.config.get_path("catalog_file")
                        catalog_path = Path(cp) if cp is not None else None
                    except Exception:
                        catalog_path = None

                if catalog_path and catalog_path.exists():
                    # parse 'time' column if present
                    cols = pd.read_csv(catalog_path, nrows=0).columns.tolist()
                    parse_dt = ["time"] if "time" in cols else None
                    orig_catalog = pd.read_csv(catalog_path, parse_dates=parse_dt)
            except Exception:
                orig_catalog = None

            # If matching applied a bbox/region filter, a filtered_template_info.csv may exist.
            # Use it to restrict the original catalog for a fair comparison plot.
            try:
                templates_dir = Path(self.config.get_path('templates_dir'))
                filtered_info_file = templates_dir / 'filtered_template_info.csv'
                filtered_orig = None
                if filtered_info_file.exists():
                    try:
                        finfo = pd.read_csv(filtered_info_file)
                        # derive bbox from filtered templates if lat/lon available
                        if {'lat', 'lon'}.issubset(finfo.columns):
                            min_lat = finfo['lat'].min()
                            max_lat = finfo['lat'].max()
                            min_lon = finfo['lon'].min()
                            max_lon = finfo['lon'].max()
                            if orig_catalog is not None and {'lat', 'lon'}.issubset(orig_catalog.columns):
                                filtered_orig = orig_catalog[
                                    orig_catalog['lat'].between(min_lat, max_lat) &
                                    orig_catalog['lon'].between(min_lon, max_lon)
                                ]
                                console.print(f"[dim]Using filtered original catalog from templates bbox: {len(filtered_orig)} events[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Warning: could not read filtered template info: {e}[/yellow]")

                # Choose which original catalog to plot: filtered if available, else full
                plot_orig = filtered_orig if (filtered_orig is not None and len(filtered_orig) > 0) else orig_catalog

                if plot_orig is not None and len(plot_orig) > 0:
                    # Apply same magnitude filter to original catalog for fair comparison
                    if min_mag > 0.0 and 'magnitude' in plot_orig.columns:
                        plot_orig = plot_orig[plot_orig['magnitude'] >= min_mag]
                    plot_catalog_comparison(plot_orig, catalog_filtered, plots_dir / "catalog_comparison")
                else:
                    console.print('[yellow]Original catalog not found or empty — skipping comparison plots.[/yellow]')
            except Exception as e:
                console.print(f'[yellow]Could not produce catalog comparison plots: {e}[/yellow]')
        except Exception as e:
            console.print(f'[yellow]Could not produce catalog comparison plots: {e}[/yellow]')
        
        # Use filtered catalog for all plots
        tpl_for_plotting = self._map_template_ids_for_plotting(catalog_filtered, tpl)
        # --- Template amplitude vs magnitude plot ---
        try:
            # Prepare a small template dataframe with corrected amplitudes
            tpl_amp_df = tpl.copy()
            if 'amplitude' in tpl_amp_df.columns and 'magnitude' in tpl_amp_df.columns:
                tpl_amp_df['tpl_amp_corr'] = self._distance_correct_amplitude_vec(
                    tpl_amp_df['amplitude'].values,
                    tpl_amp_df['lat'].values if 'lat' in tpl_amp_df.columns else np.full(len(tpl_amp_df), np.nan),
                    tpl_amp_df['lon'].values if 'lon' in tpl_amp_df.columns else np.full(len(tpl_amp_df), np.nan),
                    self.config.get('evaluation.geometrical_spreading', 1.0),
                )
                tpl_amp_df['tpl_magnitude'] = tpl_amp_df['magnitude']
                plot_template_regression(tpl_amp_df, plots_dir)
                console.print(f"[dim]Saved template amplitude vs magnitude plot to {plots_dir / 'template_regression.png'}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Template amplitude vs magnitude plot failed: {e}[/yellow]")
        plot_magnitude_vs_time(catalog_filtered, plots_dir, tpl_df=tpl_for_plotting)
        plot_cumulative_events(catalog_filtered, plots_dir)
        plot_frequency_magnitude(catalog_filtered, bval["Mc"], bval["b"], bval["a"], plots_dir)
        if self.config.get('evaluation.plot_waveform_comparisons', True):
            try:
                self.plot_detection_and_template_waveforms(
                    catalog_filtered, tpl, plots_dir / "waveform_comparisons", n_samples=50
                )
            except Exception as e:
                console.print(f'[yellow]Waveform comparison plots failed: {e}[/yellow]')

        # Template re-detection summary plot (use filtered catalog)
        try:
            mapping_file = self.config.get_path('base_dir') / self.config.get_path('templates_dir') / 'template_id_mapping.csv'
            mapping_df = pd.read_csv(mapping_file) if mapping_file.exists() else None
            plot_template_redetection(catalog_filtered, tpl, plots_dir, mapping_df=mapping_df)
            console.print(f"[dim]Saved template re-detection plot to {plots_dir / 'template_redetection.png'}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Template re-detection plot failed: {e}[/yellow]")

        # Mw vs estimated magnitude plot (use filtered catalog)
        try:
            mapping_file = self.config.get_path('base_dir') / self.config.get_path('templates_dir') / 'template_id_mapping.csv'
            mapping_df = pd.read_csv(mapping_file) if mapping_file.exists() else None
            plot_mw_vs_estimated(catalog_filtered, tpl, plots_dir, mapping_df=mapping_df)
            console.print(f"[dim]Saved Mw vs estimated magnitude plot to {plots_dir / 'mw_vs_estimated_magnitude.png'}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Mw vs estimated magnitude plot failed: {e}[/yellow]")
        
        # Summary statistics (show both full and filtered catalogs)
        console.print("\n" + "="*50)
        console.print("[bold]Evaluation Summary[/bold]")
        console.print(f"Method: {method}")
        console.print(f"Total detections (raw): {len(det)}")
        console.print(f"Unique events (full catalog): {len(catalog_full)}")
        if min_mag > 0.0:
            console.print(f"Events above M{min_mag:.1f} (analysis): {len(catalog_filtered)}")
        console.print(f"Detections with magnitude: {catalog_filtered['est_magnitude'].notna().sum()}")
        if len(catalog_filtered) > 0:
            console.print(f"Magnitude range: {catalog_filtered['est_magnitude'].min():.2f} to {catalog_filtered['est_magnitude'].max():.2f}")
            console.print(f"Mean magnitude: {catalog_filtered['est_magnitude'].mean():.2f} ± {catalog_filtered['est_magnitude'].std():.2f}")
            console.print(f"b-value: {bval['b']:.2f} (Mc={bval['Mc']:.2f}, a={bval['a']:.2f})")
        else:
            console.print("[yellow]No events available for magnitude statistics[/yellow]")
        console.print("="*50)
        
        return {
            "success": True,
            "method": method,
            "detections_raw": len(det),
            "detections": len(catalog_full),  # Full catalog for compatibility
            "detections_analysis": len(catalog_filtered),  # Filtered catalog used for analysis
            "detections_with_mag": int(catalog_filtered["est_magnitude"].notna().sum()),
            "magnitude_mean": float(catalog_filtered["est_magnitude"].mean()) if len(catalog_filtered) > 0 else np.nan,
            "magnitude_std": float(catalog_filtered["est_magnitude"].std()) if len(catalog_filtered) > 0 else np.nan,
            "Mc": bval["Mc"],
            "b_value": bval["b"],
            "a_value": bval["a"],
            "output_file": csv_file,
            "output_file_multi": csv_multi,
            "plots_dir": plots_dir,
            "trace": trace if use_bayesian else None
        }