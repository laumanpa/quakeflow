"""
Plotting utilities for QuakeFlow.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from obspy import read
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import linregress

def name2time(event_name: str) -> pd.Timestamp:
    """Convert event name (MOED_20250306_042229) to pandas Timestamp."""
    date_str = event_name.split("_")[1]  # '20250306'
    time_str = event_name.split("_")[2]  # '042229'
    dt_str = f"{date_str} {time_str}"
    return pd.to_datetime(dt_str, format="%Y%m%d %H%M%S", utc=True)


def plot_magnitude_vs_time(df: pd.DataFrame, output_dir: Path, tpl_df: pd.DataFrame = None):
    """Plot magnitude over time."""
    if tpl_df is not None:
        tpl_df["time"] = tpl_df["event_name"].apply(name2time)
    used_tpl_ids = df["template_id"].dropna().unique().astype(int)
    if tpl_df is not None:
        tpl_df = tpl_df[tpl_df["template_id"].isin(used_tpl_ids)]
    plt.figure(figsize=(10, 4))
    plt.scatter(tpl_df["time"], tpl_df["magnitude"], s=15, c="gray", alpha=0.4, label="Templates") if tpl_df is not None else None
    plt.scatter(df["time"], df["est_magnitude"], s=10, edgecolors="black", facecolors="none", label="Detected events")
    plt.xlabel("Time")
    plt.ylabel("Estimated Magnitude")
    plt.title("Magnitude over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_vs_time.png", dpi=150)
    plt.close()


def plot_cumulative_events(df: pd.DataFrame, output_dir: Path):
    """Plot cumulative number of events."""
    df_sorted = df.sort_values("time")
    cum_counts = np.arange(1, len(df_sorted) + 1)
    
    plt.figure(figsize=(10, 4))
    plt.step(df_sorted["time"], cum_counts, where="post", color="black")
    plt.xlabel("Time")
    plt.ylabel("Cumulative number of events")
    plt.title("Cumulative events")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_events.png", dpi=150)
    plt.close()


def plot_frequency_magnitude(df: pd.DataFrame, Mc: float, b: float, a: float, output_dir: Path):
    """Plot frequency-magnitude distribution."""
    mags = df["est_magnitude"].dropna().astype(float)

    plt.figure(figsize=(6, 5))
    if mags.empty:
        # Nothing to plot
        plt.close()
        return

    min_mag = mags.min()
    max_mag = mags.max()
    if not np.isfinite(min_mag) or not np.isfinite(max_mag):
        plt.close()
        return

    # If all magnitudes are identical, provide a small range for bins
    if min_mag == max_mag:
        bins = np.arange(np.floor(min_mag) - 0.1, np.ceil(max_mag) + 0.1, 0.1)
    else:
        bins = np.arange(np.floor(min_mag), np.ceil(max_mag) + 0.1, 0.1)

    plt.hist(mags, bins=bins, color="lightblue", edgecolor="k", alpha=0.7, label="Detected events")
    plt.axvline(Mc, color="red", linestyle="--", label=f"Mc = {Mc:.2f}")

    # Gutenberg-Richter line
    m_vals = np.linspace(Mc, max_mag + 0.5, 50)
    N = 10 ** (a - b * m_vals)
    plt.plot(m_vals, N, "k--", label=f"GR fit b={b:.2f}")

    plt.yscale("log")
    plt.xlabel("Magnitude")
    plt.ylabel("Number of events")
    plt.title("Frequency-Magnitude Distribution")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "fmd.png", dpi=150)
    plt.close()


def plot_template_regression(df: pd.DataFrame, output_dir: Path):
    """Plot template magnitude regression."""
    valid = df["tpl_amp_corr"].notna() & df["tpl_magnitude"].notna()
    if not valid.any():
        return
    
    plt.figure(figsize=(6, 4))
    plt.scatter(
        np.log10(df.loc[valid, "tpl_amp_corr"]), 
        df.loc[valid, "tpl_magnitude"], 
        c="blue", 
        label="Templates"
    )
    
    slope, intercept = linregress(
        np.log10(df.loc[valid, "tpl_amp_corr"]), 
        df.loc[valid, "tpl_magnitude"]
    )[:2]
    
    x_vals = np.linspace(
        np.log10(df.loc[valid, "tpl_amp_corr"].min()), 
        np.log10(df.loc[valid, "tpl_amp_corr"].max()), 
        50
    )
    y_vals = intercept + slope * x_vals
    
    plt.plot(x_vals, y_vals, "r--", label=f"Fit: slope={slope:.2f}")
    plt.xlabel("log10(Amplitude corrected)")
    plt.ylabel("Template magnitude")
    plt.title("Template magnitude regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "template_regression.png", dpi=150)
    plt.close()


def plot_mc_qc(df: pd.DataFrame, Mc: float, b: float, a: float, output_dir: Path):
    """Plot completeness magnitude quality control."""
    mags = df["est_magnitude"].dropna().astype(float)
    if mags.empty:
        return

    mags_above = mags[mags >= Mc]

    if mags_above.empty:
        return

    max_mag = mags.max()
    hist, bin_edges = np.histogram(mags_above, bins=np.arange(Mc, max_mag + 0.1, 0.1))
    cum_obs = np.cumsum(hist[::-1])[::-1]
    cum_pred = 10 ** (a - b * bin_edges[:-1])
    
    plt.figure(figsize=(6, 4))
    plt.plot(bin_edges[:-1], cum_obs, "o", label="Observed cumulative")
    plt.plot(bin_edges[:-1], cum_pred, "--", label="Predicted GR")
    plt.yscale("log")
    plt.xlabel("Magnitude")
    plt.ylabel("Cumulative number of events")
    plt.title("Mc / b-value QC")
    plt.legend()
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / "mc_b_qc.png", dpi=150)
    plt.close()


def plot_clusters_time(df: pd.DataFrame, output_dir: Path):
    """Plot event clusters over time."""
    if "template_id" not in df.columns:
        return
    
    plt.figure(figsize=(10, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, df["template_id"].nunique())))
    
    for i, (tid, grp) in enumerate(df.groupby("template_id")):
        color = colors[i % len(colors)]
        plt.scatter(grp["time"], grp["est_magnitude"], s=10, alpha=0.7, 
                   color=color, label=f"Template {tid}")
    
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title("Event clusters over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_time.png", dpi=150)
    plt.close()


def plot_clusters_map(df: pd.DataFrame, output_dir: Path):
    """Plot event clusters on map."""
    if "lat" not in df.columns or df["lat"].isnull().all():
        return
    
    plt.figure(figsize=(6, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, df["template_id"].nunique())))
    
    for i, (tid, grp) in enumerate(df.groupby("template_id")):
        color = colors[i % len(colors)]
        plt.scatter(grp["lon"], grp["lat"], s=10, alpha=0.7, 
                   color=color, label=f"Template {tid}")
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Event clusters map")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_map.png", dpi=150)
    plt.close()


def plot_catalog_comparison(orig_df: pd.DataFrame, new_df: pd.DataFrame, output_dir: Path):
    """Compare original catalog and newly generated catalog.

    Produces:
    - `catalog_map_comparison.png`: spatial scatter of events (orig vs new)
    - `catalog_magnitude_hist.png`: magnitude histograms overlaid
    - `catalog_time_comparison.png`: cumulative event curves over time
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize time columns: accept 'time' or 'event_time'
    def _ensure_time_col(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        d = df.copy()
        if 'time' not in d.columns and 'event_time' in d.columns:
            # event_time may be obspy.UTCDateTime or string
            d['time'] = d['event_time'].apply(lambda t: pd.to_datetime(str(t)) if pd.notna(t) else pd.NaT)
        else:
            d['time'] = pd.to_datetime(d['time'], utc=True, errors='coerce')
        return d

    o = _ensure_time_col(orig_df)
    n = _ensure_time_col(new_df)

    # --- Map comparison
    try:
        if 'lat' in o.columns and 'lon' in o.columns and 'lat' in n.columns and 'lon' in n.columns:
            plt.figure(figsize=(8, 6))
            plt.scatter(o['lon'], o['lat'], s=10, c='blue', alpha=0.4, label='Original')
            plt.scatter(n['lon'], n['lat'], s=10, c='orange', alpha=0.6, label='New')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Catalog: Original vs New (spatial)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / 'catalog_map_comparison.png', dpi=150)
            plt.close()
    except Exception:
        pass

    # --- Magnitude histogram comparison
    try:
        mags_o = pd.to_numeric(o.get('magnitude', pd.Series(dtype=float)), errors='coerce').dropna()
        mags_n = pd.to_numeric(n.get('magnitude', pd.Series(dtype=float)), errors='coerce').dropna()
        if not mags_o.empty or not mags_n.empty:
            plt.figure(figsize=(8, 5))
            bins = None
            if not mags_o.empty and not mags_n.empty:
                mn = min(mags_o.min(), mags_n.min())
                mx = max(mags_o.max(), mags_n.max())
                bins = np.arange(np.floor(mn), np.ceil(mx) + 0.1, 0.1)
            elif not mags_o.empty:
                bins = np.arange(np.floor(mags_o.min()), np.ceil(mags_o.max()) + 0.1, 0.1)
            elif not mags_n.empty:
                bins = np.arange(np.floor(mags_n.min()), np.ceil(mags_n.max()) + 0.1, 0.1)

            if bins is None:
                bins = 10

            if not mags_o.empty:
                plt.hist(mags_o, bins=bins, alpha=0.5, label='Original', color='blue', edgecolor='k')
            if not mags_n.empty:
                plt.hist(mags_n, bins=bins, alpha=0.5, label='New', color='orange', edgecolor='k')
            plt.xlabel('Magnitude')
            plt.ylabel('Count')
            plt.title('Magnitude distribution: Original vs New')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / 'catalog_magnitude_hist.png', dpi=150)
            plt.close()
    except Exception:
        pass

    # --- Cumulative time comparison
    try:
        if 'time' in o.columns and 'time' in n.columns:
            o_sorted = o.dropna(subset=['time']).sort_values('time')
            n_sorted = n.dropna(subset=['time']).sort_values('time')
            plt.figure(figsize=(10, 4))
            if len(o_sorted) > 0:
                plt.step(o_sorted['time'], np.arange(1, len(o_sorted) + 1), where='post', label='Original', color='blue')
            if len(n_sorted) > 0:
                plt.step(n_sorted['time'], np.arange(1, len(n_sorted) + 1), where='post', label='New', color='orange')
            plt.xlabel('Time')
            plt.ylabel('Cumulative events')
            plt.title('Cumulative events over time: Original vs New')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(output_dir / 'catalog_time_comparison.png', dpi=150)
            plt.close()
    except Exception:
        pass

def _template_to_vector(trace, nfft: int = 1024):
    data = trace.data.astype(np.float32)
    data = data - (data.mean() if np.isfinite(data.mean()) else 0.0)
    std = np.std(data)
    if std > 0:
        data = data / std
    if len(data) < nfft:
        data = np.pad(data, (0, nfft - len(data)))
    else:
        data = data[:nfft]
    spec = np.abs(np.fft.rfft(data))
    norm = np.linalg.norm(spec) + 1e-12
    return spec / norm

def plot_templates_tsne(templates_dir: Path, output_dir: Path,
                        eps: float = 0.2,
                        perplexity: int = 30,
                        n_iter: int = 1000,
                        random_state: int = 42,
                        nfft: int = 1024,
                        filename: str = "templates_tsne.png"):
    """Visualize template clustering with t-SNE in 2D.

    - Computes simple spectral feature vectors from Z component of templates
    - Clusters with DBSCAN (cosine distance) to color points
    - Embeds with t-SNE and saves `templates_tsne.png`
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(templates_dir).glob("*.mseed"))
    if len(files) == 0:
        return

    vectors = []
    labels = []
    names = []

    for f in files:
        try:
            st = read(str(f))
            z = st.select(channel="*Z")
            if len(z) == 0:
                # fallback to first trace
                tr = st[0]
            else:
                tr = z[0]
            vec = _template_to_vector(tr, nfft=nfft)
            vectors.append(vec)
            labels.append(f.stem)
            names.append(f.name)
        except Exception:
            continue

    if len(vectors) < 2:
        return

    X = np.vstack(vectors)
    # Cluster with cosine distances for coloring
    dist = cosine_distances(X)
    clustering = DBSCAN(eps=eps, min_samples=1, metric="precomputed").fit(dist)
    cluster_ids = clustering.labels_

    # Safe perplexity
    safe_perp = max(5, min(perplexity, (len(X) - 1) // 3 if len(X) > 1 else 5))
    # Use broadly compatible TSNE args (omit n_iter, 'auto' learning rate)
    tsne = TSNE(n_components=2, perplexity=safe_perp,
                random_state=random_state, init="pca")
    emb = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.tab20
    colors = cmap(np.linspace(0, 1, min(20, (cluster_ids.max() + 1) if cluster_ids.size else 1)))
    for i in range(len(emb)):
        c = colors[cluster_ids[i] % len(colors)] if cluster_ids[i] >= 0 else (0.5, 0.5, 0.5, 0.7)
        plt.scatter(emb[i, 0], emb[i, 1], s=20, color=c, alpha=0.9)

    # Optional: annotate a few points to aid interpretation
    for i in range(min(10, len(emb))):
        plt.text(emb[i, 0], emb[i, 1], str(i), fontsize=7, alpha=0.7)

    plt.title("Template Clusters (t-SNE)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()

def plot_templates_tsne_with_mapping(templates_dir: Path,
                                     mapping_file: Path,
                                     output_dir: Path,
                                     perplexity: int = 30,
                                     n_iter: int = 1000,
                                     random_state: int = 42,
                                     nfft: int = 1024):
    """Visualize templates with t-SNE using matcher cluster mapping.

    Colors are the representative cluster IDs from `template_id_mapping.csv`.
    Representative (medoid) templates are highlighted with larger markers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(templates_dir).glob("*.mseed"))
    if len(files) == 0:
        return

    # Load mapping
    if not Path(mapping_file).exists():
        # Fallback to plain tsne without mapping
        return plot_templates_tsne(templates_dir, output_dir,
                                   eps=0.2, perplexity=perplexity,
                                   n_iter=n_iter, random_state=random_state,
                                   nfft=nfft)

    try:
        if Path(mapping_file).stat().st_size == 0:
            return plot_templates_tsne(templates_dir, output_dir,
                                       eps=0.2, perplexity=perplexity,
                                       n_iter=n_iter, random_state=random_state,
                                       nfft=nfft)
        map_df = pd.read_csv(mapping_file)
    except Exception:
        return plot_templates_tsne(templates_dir, output_dir,
                                   eps=0.2, perplexity=perplexity,
                                   n_iter=n_iter, random_state=random_state,
                                   nfft=nfft)
    # Expect columns: original_template_id, representative_id
    if not set(["original_template_id", "representative_id"]).issubset(map_df.columns):
        return plot_templates_tsne(templates_dir, output_dir,
                                   eps=0.2, perplexity=perplexity,
                                   n_iter=n_iter, random_state=random_state,
                                   nfft=nfft)

    # Build features
    vectors = []
    for f in files:
        try:
            st = read(str(f))
            z = st.select(channel="*Z")
            tr = z[0] if len(z) else st[0]
            vectors.append(_template_to_vector(tr, nfft=nfft))
        except Exception:
            # Keep alignment by inserting a zero vector
            vectors.append(np.zeros(nfft//2 + 1, dtype=np.float32))

    if len(vectors) < 2:
        return

    X = np.vstack(vectors)

    # Safe perplexity
    safe_perp = max(5, min(perplexity, (len(X) - 1) // 3 if len(X) > 1 else 5))
    tsne = TSNE(n_components=2, perplexity=safe_perp,
                random_state=random_state, init="pca")
    emb = tsne.fit_transform(X)

    # Color by representative cluster ID
    # mapping row original_template_id corresponds to index in sorted files
    color_ids = np.full(len(files), -1, dtype=int)
    rep_indices = set()
    for _, row in map_df.iterrows():
        orig = int(row["original_template_id"]) if pd.notna(row["original_template_id"]) else None
        rep = int(row["representative_id"]) if pd.notna(row["representative_id"]) else None
        if orig is None or rep is None:
            continue
        if 0 <= orig < len(files):
            color_ids[orig] = rep
            # representative_id points to the cluster ID, highlight any one medoid index
            rep_indices.add(orig) if rep == color_ids[orig] else None

    plt.figure(figsize=(8, 6))
    unique_ids = np.unique(color_ids[color_ids >= 0])
    cmap = plt.cm.tab20
    colors = {cid: cmap((i % 20) / 20.0) for i, cid in enumerate(unique_ids)}

    for i in range(len(emb)):
        cid = color_ids[i]
        c = colors.get(cid, (0.5, 0.5, 0.5, 0.7))
        size = 30 if i in rep_indices else 18
        edge = 'k' if i in rep_indices else None
        plt.scatter(emb[i, 0], emb[i, 1], s=size, c=[c], alpha=0.9, edgecolors=edge, linewidths=0.6 if edge else 0)

    plt.title("Template Clusters (t-SNE, matcher mapping)")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "templates_tsne_match.png", dpi=150)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def qc_template_vs_detected(df, outdir: Path):
    fig, ax = plt.subplots()
    ax.scatter(df["tpl_magnitude"], df["est_magnitude"],
               s=8, alpha=0.5)

    mmin = min(df["tpl_magnitude"].min(), df["est_magnitude"].min())
    mmax = max(df["tpl_magnitude"].max(), df["est_magnitude"].max())
    ax.plot([mmin, mmax], [mmin, mmax], "k--", lw=1)

    ax.set_xlabel("Template Magnitude")
    ax.set_ylabel("Detected Magnitude")
    ax.set_title("QC: Template vs Detected Magnitude")
    ax.grid(alpha=0.3)

    fig.savefig(outdir / "qc_template_vs_detected.png", dpi=200)
    plt.close(fig)

def qc_residual_vs_similarity(df, outdir: Path):
    residual = df["est_magnitude"] - df["tpl_magnitude"]

    fig, ax = plt.subplots()
    ax.scatter(df["similarity"], residual,
               s=8, alpha=0.5)

    ax.axhline(0, color="k", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Similarity (CC)")
    ax.set_ylabel("Magnitude Residual")
    ax.set_title("QC: Residual vs Similarity")
    ax.grid(alpha=0.3)

    fig.savefig(outdir / "qc_residual_vs_similarity.png", dpi=200)
    plt.close(fig)

def qc_magnitude_vs_similarity(df, outdir: Path):
    fig, ax = plt.subplots()
    ax.scatter(df["similarity"], df["est_magnitude"],
               s=8, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Similarity (CC)")
    ax.set_ylabel("Estimated Magnitude")
    ax.set_title("QC: Magnitude vs Similarity")
    ax.grid(alpha=0.3)

    fig.savefig(outdir / "qc_magnitude_vs_similarity.png", dpi=200)
    plt.close(fig)

def qc_residual_histogram(df, outdir: Path):
    residual = df["est_magnitude"] - df["tpl_magnitude"]

    fig, ax = plt.subplots()
    ax.hist(residual, bins=40, density=True, alpha=0.7)

    ax.set_xlabel("Magnitude Residual")
    ax.set_ylabel("Density")
    ax.set_title("QC: Residual Distribution")
    ax.grid(alpha=0.3)

    fig.savefig(outdir / "qc_residual_histogram.png", dpi=200)
    plt.close(fig)

def qc_time_magnitude(df, outdir: Path):
    fig, ax = plt.subplots()
    ax.scatter(df["time"], df["est_magnitude"],
               s=6, alpha=0.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title("QC: Magnitude vs Time")
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    fig.savefig(outdir / "qc_magnitude_vs_time.png", dpi=200)
    plt.close(fig)

def plot_bayes_qc(df, trace, outdir):
    outdir.mkdir(exist_ok=True)

    # Posterior predictive regression
    alpha_post = trace.posterior["alpha"].mean().item()
    beta_post = trace.posterior["beta"].mean().item()
    sigma_post = trace.posterior["sigma"].mean().item()

    valid = df["est_magnitude_bayes"].notna()
    log_amp = np.log10(df.loc[valid, "event_amplitude"])
    tpl_mag = df.loc[valid, "tpl_magnitude"]
    est_mag = df.loc[valid, "est_magnitude_bayes"]
    
    # --- Regression plot
    plt.figure(figsize=(6,5))
    plt.scatter(log_amp, tpl_mag, s=8, alpha=0.5, label="Template Magnitudes")
    plt.plot(log_amp, est_mag, 'r.', alpha=0.5, label="Bayesian estimate")
    xfit = np.linspace(log_amp.min(), log_amp.max(), 100)
    plt.plot(xfit, alpha_post*xfit + beta_post, 'k--', lw=1, label="Posterior mean")
    plt.fill_between(xfit, alpha_post*xfit + beta_post - sigma_post,
                     alpha_post*xfit + beta_post + sigma_post,
                     color='orange', alpha=0.2, label="Posterior 1σ")
    plt.xlabel("log10(Event amplitude)")
    plt.ylabel("Template Magnitude")
    plt.legend()
    plt.title("QC: Bayesian Magnitude Regression")
    plt.grid(alpha=0.3)
    plt.savefig(outdir / "qc_bayes_magnitude.png", dpi=200)
    plt.close()

def plot_waveform_comparison(template_trace, detected_trace, outdir: Path, event_id: str, similarity: float):
    outdir.mkdir(exist_ok=True)
    if template_trace is None or detected_trace is None:
        return
    if len(template_trace) == 0 or len(detected_trace) == 0:
        return
    template_trace = template_trace[0]
    detected_trace = detected_trace[0]
    tpl_max = np.max(np.abs(template_trace.data))
    det_max = np.max(np.abs(detected_trace.data))
    if tpl_max == 0 or det_max == 0:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(template_trace.times(), template_trace.data / tpl_max, label="Template", color="blue")
    plt.plot(detected_trace.times(), detected_trace.data / det_max, label="Detected", color="orange", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.title(f"Waveform Comparison for Event {event_id} (Similarity: {similarity:.2f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"waveform_comparison_{event_id}.png", dpi=200)
    plt.close()


def plot_template_redetection(catalog_df: pd.DataFrame,
                              template_info_df: pd.DataFrame,
                              output_dir: Path,
                              mapping_df: pd.DataFrame = None):
    """Plot how many templates were re-detected vs total templates.

    Produces a two-panel figure:
      Left  – horizontal bar chart: detected (green) vs undetected (red) count.
      Right – bar chart with detection count per template, sorted descending.

    Parameters
    ----------
    catalog_df : DataFrame
        Final detection catalog (must contain ``template_id`` or
        ``original_template_id``).
    template_info_df : DataFrame
        Template info CSV (must contain ``template_id``).
    output_dir : Path
        Directory where the plot is saved.
    mapping_df : DataFrame, optional
        Template ID mapping with ``original_template_id`` and
        ``representative_id`` columns.  Used to map back detections to
        original templates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(template_info_df)
    if n_total == 0:
        return

    # Determine which original template IDs produced detections
    if mapping_df is not None and 'original_template_id' in mapping_df.columns:
        # Map detection template_ids back to all original template IDs
        det_repr_ids = set(catalog_df['template_id'].dropna().unique())
        detected_orig_ids = set(
            mapping_df.loc[
                mapping_df['representative_id'].isin(det_repr_ids),
                'original_template_id'
            ].values
        )
    elif 'original_template_id' in catalog_df.columns:
        detected_orig_ids = set(catalog_df['original_template_id'].dropna().unique())
    else:
        detected_orig_ids = set(catalog_df['template_id'].dropna().unique())

    all_tpl_ids = set(template_info_df['template_id'].values)
    n_detected = len(detected_orig_ids & all_tpl_ids)
    n_undetected = n_total - n_detected

    # Count detections per original template
    if 'original_template_id' in catalog_df.columns:
        det_counts = catalog_df['original_template_id'].value_counts()
    else:
        det_counts = catalog_df['template_id'].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 2.5]})

    # --- Left panel: detected vs undetected summary ---
    ax = axes[0]
    bars = ax.barh(['Detected', 'Undetected'], [n_detected, n_undetected],
                   color=['#2ecc71', '#e74c3c'], edgecolor='k')
    ax.bar_label(bars, fmt='%d', fontsize=11)
    ax.set_xlabel('Number of templates')
    ax.set_title(f'Template Re-detection\n({n_detected}/{n_total} = {100*n_detected/n_total:.0f}%)')
    ax.set_xlim(0, max(n_detected, n_undetected) * 1.25)
    ax.grid(axis='x', alpha=0.3)

    # --- Right panel: detections per template ---
    ax2 = axes[1]
    if len(det_counts) > 0:
        # Limit to top 50 templates for readability
        top_n = min(50, len(det_counts))
        top = det_counts.head(top_n)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
        ax2.bar(range(len(top)), top.values, color=colors, edgecolor='k', linewidth=0.3)
        ax2.set_xlabel('Template (sorted by detections)')
        ax2.set_ylabel('Number of detections')
        ax2.set_title(f'Detections per Template (top {top_n})')
        ax2.set_xticks([])
        ax2.grid(axis='y', alpha=0.3)

        # Annotate the most prolific template
        ax2.annotate(f'Tpl {top.index[0]}\n({top.values[0]})',
                     xy=(0, top.values[0]),
                     xytext=(3, top.values[0] * 1.1),
                     fontsize=8, ha='left',
                     arrowprops=dict(arrowstyle='->', color='gray'))
    else:
        ax2.text(0.5, 0.5, 'No detections', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14, color='gray')

    fig.tight_layout()
    fig.savefig(output_dir / 'template_redetection.png', dpi=200)
    plt.close(fig)


def plot_mw_vs_estimated(catalog_df: pd.DataFrame,
                         template_info_df: pd.DataFrame,
                         output_dir: Path,
                         mapping_df: pd.DataFrame = None):
    """Plot catalog Mw vs estimated magnitude for self-detections.

    For each detection that corresponds to a known template event, plot
    the catalog magnitude (Mw) on the x-axis against the estimated
    magnitude on the y-axis.  Adds a 1:1 reference line and a linear
    best-fit.

    Parameters
    ----------
    catalog_df : DataFrame
        Final detection catalog with ``est_magnitude`` and ``template_id``
        (or ``original_template_id``).
    template_info_df : DataFrame
        Template info with ``template_id`` and ``magnitude`` columns.
    output_dir : Path
        Directory where the plot is saved.
    mapping_df : DataFrame, optional
        Mapping with ``original_template_id`` ↔ ``representative_id``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a lookup: original_template_id → catalog magnitude
    if 'magnitude' not in template_info_df.columns:
        return
    tpl_mag = template_info_df.set_index('template_id')['magnitude'].to_dict()

    # Resolve detection template_id → original_template_id → catalog mag
    df = catalog_df.copy()
    if mapping_df is not None and 'original_template_id' in mapping_df.columns:
        repr_to_orig = (
            mapping_df.groupby('representative_id')['original_template_id']
            .apply(list).to_dict()
        )
        # For detections: use original_template_id if available
        if 'original_template_id' in df.columns:
            df['catalog_mw'] = df['original_template_id'].map(tpl_mag)
        else:
            # Map via representative → first original that has a magnitude
            def _first_mag(repr_id):
                for oid in repr_to_orig.get(repr_id, []):
                    m = tpl_mag.get(oid)
                    if m is not None and np.isfinite(m):
                        return m
                return np.nan
            df['catalog_mw'] = df['template_id'].apply(_first_mag)
    elif 'original_template_id' in df.columns:
        df['catalog_mw'] = df['original_template_id'].map(tpl_mag)
    else:
        df['catalog_mw'] = df['template_id'].map(tpl_mag)

    # Also try using 'magnitude' or 'tpl_magnitude' columns directly if present
    if 'tpl_magnitude' in df.columns:
        mask = df['catalog_mw'].isna()
        df.loc[mask, 'catalog_mw'] = df.loc[mask, 'tpl_magnitude']
    elif 'magnitude' in df.columns:
        mask = df['catalog_mw'].isna()
        df.loc[mask, 'catalog_mw'] = df.loc[mask, 'magnitude']

    # Filter to rows with both values
    valid = df[['catalog_mw', 'est_magnitude']].dropna()
    if len(valid) < 2:
        return

    mw = valid['catalog_mw'].values.astype(float)
    est = valid['est_magnitude'].values.astype(float)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Scatter
    sim_col = valid['similarity'].values if 'similarity' in valid.columns else None
    if sim_col is not None:
        sc = ax.scatter(mw, est, s=12, alpha=0.5, c=sim_col,
                        cmap='viridis', edgecolors='k', linewidth=0.3)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('CC similarity')
    else:
        ax.scatter(mw, est, s=12, alpha=0.5, c='steelblue',
                   edgecolors='k', linewidth=0.3)

    # 1:1 line
    lo = min(mw.min(), est.min()) - 0.2
    hi = max(mw.max(), est.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='1:1')

    # Linear fit
    try:
        slope, intercept, r_value, _, _ = linregress(mw, est)
        x_fit = np.linspace(lo, hi, 50)
        ax.plot(x_fit, slope * x_fit + intercept, 'r-', lw=1.5,
                label=f'Fit: y={slope:.2f}x{intercept:+.2f} (R²={r_value**2:.2f})')
    except Exception:
        pass

    # Residual stats
    residual = est - mw
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    ax.text(0.05, 0.95, f'N = {len(mw)}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_xlabel('Catalog Magnitude (Mw)')
    ax.set_ylabel('Estimated Magnitude')
    ax.set_title('Catalog Mw vs Estimated Magnitude')
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'mw_vs_estimated_magnitude.png', dpi=200)
    plt.close(fig)
