# CODE_MAP for FINAL-CODE

This file gives a compact map of the uploaded code files and their role in the DEMETER LSTM-AE analysis pipeline.

## 1. Data conversion and feature preparation

### `IMSC_ICE_ASCII.ipynb`
Converts DEMETER ICE/IMSC binary orbit files into readable dataframes. It decodes binary records, builds orbit-level data, checks single-year conversion, selects clean orbit lists, and supports batch conversion over a parent data folder.

### `GRID-data-Preperation.py`
Creates the geographic grid and extracts spectral quantile features. The script:

- reads the clean DEMETER ICE orbit list;
- loads orbit-level ASCII/pickle spectra;
- converts longitudes to the `[-180, 180]` range;
- creates 20° × 20° grid cells;
- creates local spatial polygons around orbit points;
- computes Q1, Q2, and Q3 for 11 ICE frequency bands;
- assigns each local window to the grid cell with maximum overlap;
- saves the aggregated grid dictionary as a pickle file.

## 2. Exploratory and descriptive analysis

### `Grid-Timesiers-plot.ipynb`
Plots grid-level Q3/frequency-band time series and input-data examples.

### `Data_statistics.ipynb`
Summarises data availability across rolling windows and seismic/non-seismic classes. It includes visual checks of training and validation class balance.

### `EQ_statistics.ipynb`
Plots earthquake catalogue statistics, including event counts by year.

### `Heatmap-correlation.ipynb`
Calculates and plots correlation matrices among quantiles and frequency bands.

### `Q_FB-CorrelationPlots-GRID.ipynb`
Produces grid-based quantile/frequency-band correlation plots and heatmaps.

## 3. Background correction

### `Rolling_Window_BGcorrected_Data.ipynb`
Creates rolling-window background-corrected data. Main steps:

- define 12-month training and 3-month validation windows;
- resample training data using 30-day windows;
- fit a background model using a linear trend plus annual sinusoid;
- compute Q3 residuals for each frequency band and grid cell;
- save one `Background_data-window_<i>.pkl` file per rolling window.

### `Rolling_Window-BGcorrected-Data.py`
Script version of the rolling-window background-correction notebook. Use this for HPC or terminal execution.

### `BackgroundCorrection-Plot.ipynb`
Creates an explanatory plot for the background-correction method. It shows original Q3 values, 30-day filtered means, fitted background, and corrected residuals for a selected grid/frequency-band/window example.

## 4. LSTM autoencoder training and evaluation

### `DEMETER_LSTM_AE-Training.ipynb`
Main rolling-window LSTM-AE training workflow. It:

- loads residual data, earthquake catalogue, storm data, and rolling-window background files;
- prepares train/validation datasets;
- trains the LSTM autoencoder;
- computes reconstruction errors;
- applies percentile thresholds;
- removes storm-affected anomaly sequences where storm data are available;
- runs seismic association analysis;
- saves model files, result tables, loss curves, and reconstruction-error plots.

### `DEMETER_LSTM_AE-Training-loadpretrained.ipynb`
Similar to the main training notebook, but starts each rolling window from a previously trained model where available. This supports weight-updating or continuation across windows.

### `Retraining_Hp-B.sh`
SLURM batch script for running one hyperparameter training job on the GPU partition.

## 5. Seismic labels, random baseline, and thresholds

### `Seismic-Labelling_Summary--RDM_prep.ipynb`
Prepares seismic labels and randomised/control files. It supports the random sampling test by generating labelled random/control sequence tables for train and validation windows.

### `Random_baseline_rolling_windows.ipynb`
Builds the random baseline by sampling sequences and checking earthquake association under the same spatial/temporal criteria. This gives the reference distribution used to compare model-selected anomalies with random selection.

### `Anomaly_threshold_analysis.ipynb`
Runs threshold sensitivity analysis on already trained rolling-window models. It applies percentile thresholds from 95% to 99%, evaluates anomalies, applies storm correction, and saves threshold-sweep result tables.

## 6. Final result plotting and model selection

### `Results_PLOT_SW22TW48.ipynb`
Creates final SW22 × TW48 result figures. It includes train/validation ratio plots, anomaly-count bars, sunspot overlays, true-positive versus ground-truth comparisons, Q-factor/Z-like summaries, and combined train-validation plots.

### `Best-Performing-Criteria.ipynb`
Combines Q-factor/model-summary CSV files and ranks models or spatial-temporal criteria. It is used to identify stronger-performing configurations based on summary scores and validation behaviour.

### `SW22TW48_leadtime.ipynb`
Computes lead-time distributions for SW22 × TW48. It extracts the first matched earthquake time for each anomaly and calculates:

```text
Δt = t_eq − t_n
```

It compares model-selected lead times with random/control lead-time distributions.

## 7. NOA / Digisonde comparison

### `NOA-Digisonde_Background-PLOT.ipynb`
Processes and plots ionosonde/digisonde background behaviour for NOA-related comparison. It produces station-level figures, flagged windows, and summary CSV files for coincident ionosonde intervals.

## Main parameters to document in papers/thesis

| Parameter | Value used in final workflow |
|---|---|
| Training window | 12 months |
| Validation window | 3 months |
| Rolling stride | 3 months |
| Background resampling | 30 days |
| Background model | linear trend + annual sinusoid |
| Main spatial/temporal case | SW22 × TW48 |
| Threshold sweep | 95%, 96%, 97%, 98%, 99% |
| Model type | LSTM autoencoder |
| Main anomaly metric | reconstruction error |
| Statistical control | random baseline / random sampling test |
| Lead-time metric | Δt = earthquake time − anomaly time |

## Practical cleanup before public Git upload

- Replace hard-coded absolute paths with a config file or environment variables.
- Remove large generated outputs from Git: `.pkl`, `.pth`, large `.csv`, figures, and logs.
- Keep raw DEMETER data outside the repository.
- Add small example inputs only if redistribution is allowed.
- Add a reproducibility note specifying software versions and whether results were produced on CPU or GPU.
