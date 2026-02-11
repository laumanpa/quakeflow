# QuakeFlow

A comprehensive template matching and magnitude estimation pipeline for seismic data analysis.

## Features

- **Template Creation**: Extract templates from catalog events with automatic onset detection
- **Template Matching**: Fast correlation-based matching with clustering to reduce redundancy
- **Magnitude Estimation**: Distance-corrected magnitude estimation with b-value calculation
- **Visualization**: Comprehensive plots for result evaluation
- **Parallel Processing**: Efficient parallel processing of continuous data

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/quakeflow.git
cd quakeflow
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```
## Examples

```bash
# 1. Initialize project
quakeflow init --config config.yaml --station GWBD --base-dir ./my_project

# 2. Create templates
quakeflow create --config config.yaml events.csv --bbox 50.0 50.5 7.5 8.5 --type generic

# 3. Run matching
quakeflow match --config config.yaml

# 4. Evaluate
quakeflow evaluate --config config.yaml

# 5. Complete pipeline
quakeflow run --config config.yaml events.csv --bbox 50.0 50.5 7.5 8.5

# 6. Check status
quakeflow status --config config.yaml
```

## Catalog Formats

QuakeFlow accepts several catalog file formats. Use the `--type`/`-t` option to tell QuakeFlow how to parse the file.

- `generic`: CSV files with a `time` column (ISO-like timestamps). Common for modern catalogs exported from databases.
- `grun`: GRUN-style catalog text where timestamps are embedded in a free-text field (special parsing applied).
- `dlf`, `dlff`, `dat`: DLF-style whitespace-delimited catalogs often shipped as `.dat` files. Expected columns (whitespace-separated):
	0. date in `YYYYMMDD` (e.g. `20180105`)
	1. time in `HHMM` or `HMM` (leading zeros optional; seconds are ignored)
	2. location or event id (ignored)
	3. magnitude
	4. latitude (decimal degrees)
	5. longitude (decimal degrees)
	6. depth (optional, km)

Example `.dat` line:
```
20180105  905  -  2.3  50.1234  7.9876  10.0
```

When invoking the CLI, you can pass the catalog file (CSV or `.dat`) and set the type, for example:

```bash
quakeflow create --config config.yaml --type dat events.dat --bbox 50.0,50.2,7.8,8.0
```

The DLF/`.dat` handler requires at least six whitespace-separated columns and will extract `event_time`, `lat`, `lon`, and `magnitude` for template creation.

## Matching Parameters and Tuning

This section explains the key parameters controlling template matching and how to choose them for robust performance.

### Core Matching
- `template_matching.similarity_threshold`: Normalized cross-correlation peak threshold used to accept detections. Higher values reduce false positives but may miss weak events; lower values increase sensitivity but risk noise triggers. Start around `0.5–0.6`; raise to `0.7–0.8` for cleaner results; lower to `0.3–0.4` for exploratory runs.
- `template_matching.distance_samples`: Minimum separation (in samples) between correlation peaks (debouncing). Choose close to the template length in samples: $\text{distance\_samples} \approx 0.8\, f_s\,(\text{pre\_event}+\text{post\_event})$, where $f_s$ is sampling rate. Too small → multiple triggers per event; too large → misses closely spaced detections.
- `template_matching.min_spike_ratio`: Artifact filter around each detection using a short window (±0.5 s). Detections with $\max|x|/\mathrm{std}(x) \ge$ `min_spike_ratio` are rejected as spikes. Typical values `5–10`. Increase to allow more sharp signals; decrease to aggressively remove spikes.

### Template Quality & Clustering
- `template_matching.min_snr`: Filters low-quality templates before clustering using SNR computed at creation. Start at `2–3`. If many templates are noisy, raise to `4–6`; if you need more coverage, lower to `1.5–2`.
- `template_matching.cluster_eps`: DBSCAN `eps` on cosine distances of template spectral features, used to group similar templates and select medoids. Smaller (`0.1–0.2`) → tighter clusters, more singletons; larger (`0.25–0.35`) → bigger clusters, more aggressive deduplication. Use the pre-filter t-SNE (`--plot-tsne`) to visually inspect cluster separation.

### Signal Processing
- `template_creation.filter_min`, `template_creation.filter_max`: Band-pass applied to both templates and continuous data. Align these with the dominant frequency band of your station and event type. Narrow bands improve SNR but can distort waveforms; broad bands increase sensitivity to noise.
- `template_creation.pre_event`, `template_creation.post_event`: Template window around detected onset. Longer windows capture more coda but increase correlation footprint (and recommended `distance_samples`). Shorter windows are compact and precise but may lose energy.
- `template_matching.pre_amplitude`, `template_matching.post_amplitude`: Amplitude measurement window around each detection used in magnitude estimation. Keep consistent with template window and main energy duration.

### Execution & Range
### Magnitude Calibration Methods
- `evaluation.calibration_method`: Choose how magnitudes are estimated from amplitudes.
	- `bayes`: Bayesian linear calibration (robust to noise with uncertainty estimates).
	- `robust`: Monotonic isotonic regression mapping `M = f(log10(A_corr))` (handles non-linear amplitude–magnitude relations).
	- `ratio`: Simple amplitude ratio `M_det = M_tpl + log10(A_det/A_tpl)`.
- `evaluation.geometrical_spreading`: Exponent `γ` for distance correction `A_corr = A × (R/R0)^γ`.
- `evaluation.reference_distance`: `R0` (km) reference distance for correction.

Use `robust` when template amplitude vs magnitude shows curved or segmented trends. Keep `γ` small (e.g., 1.0) unless local attenuation/spreading suggests otherwise.
- `template_matching.n_jobs`: Parallelism for day-wise processing. Set to number of CPU cores minus one for responsiveness. Watch I/O and memory if very large.
- `template_matching.start_date`, `template_matching.days_to_process`: Time range for matching. Narrow for trials; expand once parameters are tuned.

### Practical Tuning Workflow
1. Start conservative: `similarity_threshold=0.6–0.7`, `min_spike_ratio=8`, `cluster_eps=0.2`, `min_snr=3`.
2. Visualize clusters before filtering: `quakeflow match --config config.yaml --plot-tsne` to produce `plots/templates_tsne_prefilter.png`. Adjust `cluster_eps` so clusters look coherent without over-merging.
3. Inspect detection volume vs. quality: lower `similarity_threshold` if too few detections; raise if false positives appear in QC plots.
4. Set `distance_samples` from template length: $\text{distance\_samples}\approx0.8\,f_s\,(\text{pre}+\text{post})$. Increase if you see duplicate detections in quick succession.
5. Re-run on a few representative days. Check QC plots (magnitude vs. time, residual vs. similarity) to validate behavior.
6. Scale up the date range once satisfied, then evaluate and refine `filter_min/max` if band content looks mismatched.

### Troubleshooting Tips
- Few detections: lower `similarity_threshold`, widen band-pass, reduce `min_snr`.
- Many false positives: raise `similarity_threshold`, lower `cluster_eps` (fewer templates), decrease band width, lower `min_spike_ratio` to reject sharp artifacts.
- Duplicate triggers per event: increase `distance_samples`.
- Over-merged clusters: decrease `cluster_eps`; under-merged: increase `cluster_eps`.

### Visualization Helpers
- Pre-filter t-SNE (all templates, colored by DBSCAN):
	```bash
	quakeflow match --config config.yaml --plot-tsne
	# Outputs: plots/templates_tsne_prefilter.png
	```
- Post-match mapping t-SNE (colored by medoid clusters): saved as `plots/templates_tsne_match.png`.