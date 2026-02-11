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

import numpy as np
import pandas as pd

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
    # ---------------------------------------------------------------------
    # b-value
    # ---------------------------------------------------------------------
    def calc_bvalue(
        self, mags: np.ndarray, Mc: Optional[float] = None
    ) -> Dict[str, float]:
        """Maximum likelihood b-value estimate."""
        mags = np.asarray(mags)
        mags = mags[np.isfinite(mags)]
        if len(mags) < 5:
            return {"Mc": np.nan, "b": np.nan, "a": np.nan}

        if Mc is None:
            Mc = np.percentile(mags, 90) - 0.1

        mags_above = mags[mags >= Mc]
        if len(mags_above) < 5:
            return {"Mc": Mc, "b": np.nan, "a": np.nan}

        mean_mag = mags_above.mean()
        b = np.log10(np.e) / (mean_mag - Mc)
        a = np.log10(len(mags_above)) + b * Mc

        return {"Mc": Mc, "b": b, "a": a}
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
        """Geometrical spreading correction."""
        if not np.isfinite(amplitude):
            return np.nan

        st_lat = self.config["stations.lat"]
        st_lon = self.config["stations.lon"]
        R0 = self.config["evaluation.reference_distance"]

        dist_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, st_lat, st_lon)
        R = max(dist_m / 1000.0, R0)

        return amplitude * (R / R0) ** gamma

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
        
        # Lade Mapping wenn vorhanden
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        merge_on = "template_id"
        if mapping_file.exists():
            try:
                if mapping_file.stat().st_size == 0:
                    console.print(f"[yellow]Warning: mapping file {mapping_file} is empty — skipping mapping.[/yellow]")
                else:
                    mapping_df = pd.read_csv(mapping_file)
                    if mapping_df.empty or not {"representative_id", "original_template_id"}.issubset(mapping_df.columns):
                        console.print(f"[yellow]Warning: mapping file {mapping_file} missing required columns — skipping mapping.[/yellow]")
                    else:
                        df = df.merge(
                            mapping_df[["representative_id", "original_template_id"]],
                            left_on="template_id",
                            right_on="representative_id",
                            how="left"
                        )
                        # Verwende original_template_id für das Mergen und bereinige Doppelspalten
                        merge_on = "original_template_id"
                        if "representative_id" in df.columns:
                            df = df.drop(columns=["representative_id"])
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                console.print(f"[yellow]Warning: failed to read mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: unexpected error reading mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
        
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

        # --- distance correction
        df["tpl_amp_corr"] = df.apply(
            lambda r: self._distance_correct_amplitude(
                r.get("tpl_amplitude", np.nan), r.get("tpl_lat", np.nan), r.get("tpl_lon", np.nan), gamma
            ),
            axis=1,
        )


        # --- magnitude via amplitude ratio
        valid = (
            (df["tpl_amp_corr"] > 0)
            & (df["det_amp_corr"] > 0)
            & df["tpl_magnitude"].notna()
        )

        if valid.sum() == 0:
            raise ValueError("No valid amplitudes for magnitude estimation.")

        df.loc[valid, "est_magnitude"] = (
            df.loc[valid, "tpl_magnitude"]
            + np.log10(df.loc[valid, "det_amp_corr"]
                       / df.loc[valid, "tpl_amp_corr"])
        )

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

        # Map template IDs if clustering was applied
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        merge_on = "template_id"
        if mapping_file.exists():
            try:
                if mapping_file.stat().st_size == 0:
                    console.print(f"[yellow]Warning: mapping file {mapping_file} is empty — skipping mapping.[/yellow]")
                else:
                    mapping_df = pd.read_csv(mapping_file)
                    if mapping_df.empty or not {"representative_id", "original_template_id"}.issubset(mapping_df.columns):
                        console.print(f"[yellow]Warning: mapping file {mapping_file} missing required columns — skipping mapping.[/yellow]")
                    else:
                        df = df.merge(mapping_df[["representative_id", "original_template_id"]],
                                      left_on="template_id", right_on="representative_id", how="left")
                        df = self._normalize_mapping_columns(df)
                        merge_on = "original_template_id"
                        if "representative_id" in df.columns:
                            df = df.drop(columns=["representative_id"])
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                console.print(f"[yellow]Warning: failed to read mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: unexpected error reading mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")

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
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])
        # Prefix template fields
        for c in tpl_cols:
            if c in df.columns:
                df.rename(columns={c: f"tpl_{c}"}, inplace=True)

        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)

        def corr_amp(val, lat, lon):
            return self._distance_correct_amplitude(val, lat, lon, gamma)

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
                df[f"tpl_amp_corr{suffix}"] = df.apply(
                    lambda r: corr_amp(r.get(tpl_col, np.nan), r.get("tpl_lat", np.nan), r.get("tpl_lon", np.nan)), axis=1)
                df[f"det_amp_corr{suffix}"] = df.apply(
                    lambda r: corr_amp(r.get(det_col, np.nan), r.get("tpl_lat", np.nan), r.get("tpl_lon", np.nan)), axis=1)
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
        
        # Lade Mapping wenn vorhanden
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        merge_on = "template_id"
        if mapping_file.exists():
            try:
                if mapping_file.stat().st_size == 0:
                    console.print(f"[yellow]Warning: mapping file {mapping_file} is empty — skipping mapping.[/yellow]")
                else:
                    mapping_df = pd.read_csv(mapping_file)
                    if mapping_df.empty or not {"representative_id", "original_template_id"}.issubset(mapping_df.columns):
                        console.print(f"[yellow]Warning: mapping file {mapping_file} missing required columns — skipping mapping.[/yellow]")
                    else:
                        df = df.merge(mapping_df[["representative_id", "original_template_id"]], 
                                      left_on="template_id", right_on="representative_id", how="left")
                        # Verwende original_template_id für das Mergen und bereinige Doppelspalten
                        merge_on = "original_template_id"
                        if "representative_id" in df.columns:
                            df = df.drop(columns=["representative_id"])
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                console.print(f"[yellow]Warning: failed to read mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: unexpected error reading mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
        
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
        
        # Calculate distances for all detections
        st_lat = self.config["stations.lat"]
        st_lon = self.config["stations.lon"]
        
        df["distance_km"] = df.apply(
            lambda r: gps2dist_azimuth(r["lat"], r["lon"], st_lat, st_lon)[0] / 1000.0,
            axis=1
        )
        
        # Apply distance correction to amplitudes
        df["tpl_amp_corr"] = df["amplitude"] * (df["distance_km"] / R0) ** gamma
        df["det_amp_corr"] = df["event_amplitude"] * (df["distance_km"] / R0) ** gamma
        
        # Filter valid data
        valid = (
            (df["tpl_amp_corr"] > 0) & 
            (df["det_amp_corr"] > 0) & 
            df["magnitude"].notna()
        )
        
        if valid.sum() == 0:
            raise ValueError("No valid amplitudes for Bayesian magnitude estimation!")
        
        # Prepare data for PyMC model
        log_tpl_amp = np.log10(df.loc[valid, "tpl_amp_corr"].values)
        log_det_amp = np.log10(df.loc[valid, "det_amp_corr"].values)
        tpl_mag = df.loc[valid, "magnitude"].values
        
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
        Robust magnitude estimation when template amplitude-magnitude relation is non-linear.

        Approach:
        - Build a monotonic mapping M = f(log10(A_corr)) using template data
        - Apply the mapping to detection corrected amplitudes
        - Fallback to Huber linear regression if isotonic fit is unstable
        """

        # Prepare template calibration data
        tpl = template_info_df.copy()
        tpl = tpl.rename(columns={"amplitude": "tpl_amplitude", "magnitude": "tpl_magnitude"})

        gamma = self.config.get("evaluation.geometrical_spreading", 1.0)
        R0 = self.config.get("evaluation.reference_distance", 1.0)
        st_lat = self.config["stations.lat"]
        st_lon = self.config["stations.lon"]

        # Distance correction for templates
        def corr_amp_tpl(row):
            if not np.isfinite(row.get("tpl_amplitude", np.nan)):
                return np.nan
            lat = row.get("lat", np.nan)
            lon = row.get("lon", np.nan)
            if not (np.isfinite(lat) and np.isfinite(lon)):
                return np.nan
            d_m, _, _ = gps2dist_azimuth(lat, lon, st_lat, st_lon)
            R = max(d_m / 1000.0, R0)
            return float(row["tpl_amplitude"]) * (R / R0) ** gamma

        tpl["tpl_amp_corr"] = tpl.apply(corr_amp_tpl, axis=1)
        valid_tpl = tpl[tpl["tpl_amp_corr"].notna() & tpl["tpl_magnitude"].notna()]
        if len(valid_tpl) < 5:
            raise ValueError("Not enough valid templates for robust calibration.")

        x_tpl = np.log10(valid_tpl["tpl_amp_corr"].values)
        y_tpl = valid_tpl["tpl_magnitude"].values

        # Fit mapping
        mapping_ok = False
        try:
            if method == "isotonic":
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso.fit(x_tpl, y_tpl)
                mapping_ok = True
        except Exception:
            mapping_ok = False

        # Fallback: robust linear (Huber)
        if not mapping_ok:
            huber = HuberRegressor()
            huber.fit(x_tpl.reshape(-1, 1), y_tpl)

        # Prepare detections with corrected amplitudes
        df = detections_df.copy()

        # Map template IDs if clustering was applied
        mapping_file = self.config.get_path('base_dir') / self.config.get_path("templates_dir") / "template_id_mapping.csv"
        merge_on = "template_id"
        if mapping_file.exists():
            try:
                # Skip empty or whitespace-only files
                if mapping_file.stat().st_size == 0:
                    console.print(f"[yellow]Warning: mapping file {mapping_file} is empty — skipping mapping.[/yellow]")
                else:
                    mapping_df = pd.read_csv(mapping_file)
                    # Validate expected columns
                    if mapping_df.empty or not {"representative_id", "original_template_id"}.issubset(mapping_df.columns):
                        console.print(f"[yellow]Warning: mapping file {mapping_file} missing required columns — skipping mapping.[/yellow]")
                    else:
                        df = df.merge(mapping_df[["representative_id", "original_template_id"]],
                                      left_on="template_id", right_on="representative_id", how="left")
                        merge_on = "original_template_id"
                        if "representative_id" in df.columns:
                            df = df.drop(columns=["representative_id"])
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                console.print(f"[yellow]Warning: failed to read mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Warning: unexpected error reading mapping file {mapping_file}: {e} — skipping mapping.[/yellow]")

        df = df.merge(
            template_info_df[["template_id", "lat", "lon"]],
            left_on=merge_on,
            right_on="template_id",
            how="left",
            suffixes=("", "_tpl")
        )
        if "template_id_tpl" in df.columns:
            df = df.drop(columns=["template_id_tpl"])

        def corr_amp_det(row):
            amp = row.get("event_amplitude", np.nan)
            lat = row.get("lat", np.nan)
            lon = row.get("lon", np.nan)
            if not (np.isfinite(amp) and np.isfinite(lat) and np.isfinite(lon)):
                return np.nan
            d_m, _, _ = gps2dist_azimuth(lat, lon, st_lat, st_lon)
            R = max(d_m / 1000.0, R0)
            return float(amp) * (R / R0) ** gamma

        df["det_amp_corr"] = df.apply(corr_amp_det, axis=1)
        valid_det = df[df["det_amp_corr"].notna()]
        if len(valid_det) == 0:
            raise ValueError("No valid detections for robust magnitude estimation.")

        x_det = np.log10(valid_det["det_amp_corr"].values)
        if mapping_ok:
            y_pred = iso.predict(x_det)
        else:
            y_pred = huber.predict(x_det.reshape(-1, 1))

        df.loc[valid_det.index, "est_magnitude"] = y_pred

        console.print("✅ Robust magnitude estimation completed ({}).".format(
            "isotonic" if mapping_ok else "huber"
        ))

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
            if "cluster_id" in row and not pd.isna(row["cluster_id"]):
                name += f" (cluster {int(row['cluster_id'])})"

            time = row.get("time")
            if pd.isna(time):
                continue

            lat = row.get("lat", station_lat)
            lon = row.get("lon", station_lon)

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
        
    def simple_deduplicate_detections(self,df: pd.DataFrame, time_window_sec: float = 1.0) -> pd.DataFrame:
        """
        Simple deduplication based on time proximity.
        """
        if len(df) == 0:
            return df
        
        # Sort by time
        df_sorted = df.sort_values('time').reset_index(drop=True)
        
        # Create unique event IDs
        event_ids = []
        current_id = 0
        last_time = df_sorted.iloc[0]['time']
        
        for i, row in df_sorted.iterrows():
            time_diff = (row['time'] - last_time).total_seconds()
            
            if time_diff > time_window_sec:
                current_id += 1
            
            event_ids.append(current_id)
            last_time = row['time']
        
        df_sorted['event_id'] = event_ids
        
        # Keep only the best detection per event (highest similarity)
        deduplicated = df_sorted.loc[df_sorted.groupby('event_id')['similarity'].idxmax()]
        
        return deduplicated.reset_index(drop=True)
    
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
        
        for _, det_row in sampled_detections.iterrows():
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
                channel=self.config["stations.channels"]
            )
            detected_trace = self.load_waveforms(
                starttime=pd.to_datetime(det_row["time"], utc=True) - pd.Timedelta(seconds=self.config.get("template_matching.pre_event", 0.5)),
                endtime=pd.to_datetime(det_row["time"], utc=True) + pd.Timedelta(seconds=self.config.get("template_matching.post_event", 5.0)),
                station=self.config["stations.station_code"],
                channel=self.config["stations.channels"]
            )
            plot_waveform_comparison(
                template_trace=template_trace,
                detected_trace=detected_trace,
                outdir=output_dir,
                event_id=f"det_{int(det_row.name)}_tpl_{tpl_id}",
                similarity=det_row.get("similarity", np.nan)
            )
            
        
        console.print(f"🖼️ Waveform comparison plots saved to: [cyan]{output_dir}[/cyan]")

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

        det = self.simple_deduplicate_detections(det, time_window_sec=1.0)

        # Also compute magnitudes for all amplitude representations (ratio method)
        det_multi = self.estimate_magnitudes_all_methods(det, tpl)
        
        # Save results
        output_dir = self.config.get_path("output_dir")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_dir / f"detections_with_magnitude.csv"
        det.to_csv(csv_file, index=False, float_format="%.5f")
        csv_multi = output_dir / f"detections_with_magnitude_multi.csv"
        det_multi.to_csv(csv_multi, index=False, float_format="%.5f")
        
        # Create marker file
        self.write_pyrocko_marker_file(det, output_dir / f"detections.markers")

        
        
        # Calculate b-value
        bval = self.calc_bvalue(det["est_magnitude"].values)
        
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

            new_catalog = det
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
                    plot_catalog_comparison(plot_orig, new_catalog, plots_dir / "catalog_comparison")
                else:
                    console.print('[yellow]Original catalog not found or empty — skipping comparison plots.[/yellow]')
            except Exception as e:
                console.print(f'[yellow]Could not produce catalog comparison plots: {e}[/yellow]')
        except Exception as e:
            console.print(f'[yellow]Could not produce catalog comparison plots: {e}[/yellow]')
        
        tpl_for_plotting = self._map_template_ids_for_plotting(det, tpl)
        plot_magnitude_vs_time(det, plots_dir, tpl_df=tpl_for_plotting)
        plot_cumulative_events(det, plots_dir)
        plot_frequency_magnitude(det, bval["Mc"], bval["b"], bval["a"], plots_dir)
        self.plot_detection_and_template_waveforms(
            det, tpl, plots_dir / "waveform_comparisons", n_samples=50
        )
        
        # Summary statistics
        console.print("\n" + "="*50)
        console.print("[bold]Evaluation Summary[/bold]")
        console.print(f"Method: {method}")
        console.print(f"Total detections: {len(det)}")
        console.print(f"Detections with magnitude: {det['est_magnitude'].notna().sum()}")
        console.print(f"Magnitude range: {det['est_magnitude'].min():.2f} to {det['est_magnitude'].max():.2f}")
        console.print(f"Mean magnitude: {det['est_magnitude'].mean():.2f} ± {det['est_magnitude'].std():.2f}")
        console.print(f"b-value: {bval['b']:.2f} (Mc={bval['Mc']:.2f}, a={bval['a']:.2f})")
        console.print("="*50)
        
        return {
            "success": True,
            "method": method,
            "detections": len(det),
            "detections_with_mag": det["est_magnitude"].notna().sum(),
            "magnitude_mean": float(det["est_magnitude"].mean()),
            "magnitude_std": float(det["est_magnitude"].std()),
            "Mc": bval["Mc"],
            "b_value": bval["b"],
            "a_value": bval["a"],
            "output_file": csv_file,
            "output_file_multi": csv_multi,
            "plots_dir": plots_dir,
            "trace": trace if use_bayesian else None
        }