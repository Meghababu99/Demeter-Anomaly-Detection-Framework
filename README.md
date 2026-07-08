# DEMETER LSTM-AE Seismo-Ionospheric Anomaly Detection

This repository contains the code structure used to rebuild and reproduce the PhD study on detecting anomalous ionospheric electric-field perturbations from DEMETER ICE satellite observations and evaluating their statistical association with global earthquakes.

The project is organized around an unsupervised LSTM autoencoder workflow. The model is trained on nominal/non-seismic DEMETER half-orbit sequences, detects anomalies through reconstruction error, and evaluates whether detected anomalies are spatially and temporally associated with earthquake events.

> **Scientific scope**  
> This repository supports a statistical anomaly-detection and seismic-association study. It does not claim deterministic earthquake prediction or direct physical causation. The detected events should be interpreted as candidate ionospheric anomalies whose association with seismicity is evaluated statistically.

---

## Repository structure

```text
Demeter-/
│
├── lstm/
│   ├── __init__.py
│   ├── dataset.py
│   ├── scaling.py
│   ├── models.py
│   ├── training.py
│   ├── training_modes.py
│   ├── training_classifier.py
│   ├── anomaly_detection.py
│   ├── SeismicCriteria.py
│   ├── Threshold_anomaly.py
│   ├── Threshold_anomaly_WOG.py # Specific for criteria applied naming WOG - Not used in final thesis 
│   ├── timewindow_analysis.py #Not used in the current analysis
│   ├── fb_analyser.py #Not used in the current analysis
│   ├── classifier_evaluation.py
│   ├── AnomalyAnalyser.py
│   └── out_encoder.py
│
├── Final-Code/
│   ├── *.ipynb
│   └── *.py
│
├── Models-Trained/
│   ├──  SW22_TW48/
│   └── ......
│
├── Data/
│   ├── storm_data.pkl
│   ├── Down_Orbits-Q3-loc-Train-limitedgrids_RES.pkl
│   ├── Main_earthquakes.csv
│   ├── Bg_window_data/
│   │   ├── Background_data-window_0.pkl
│   │   ├── Background_data-window_1.pkl
│   │   └── ...
│   └── RST-Lable/
│       ├── RDM_df_train_30D-5SW-tw48_w0.csv
│   └── Label_data/
│       ├── summary_df_train_30D-22SW-tw48_w0.csv
│       ├── summary_df_val_30D-22SW-tw48_w0.csv
│       └── ...
│
├── Result Files/
├   ├──SW22_TW48/
│   │   ├── *.csv
│   │   └── figures/
│   │   
│   └── Z-factor-files/
│       ├── *.csv
│
├── README.md
└── requirements.txt
```

---

## Folder descriptions

### `lstm/`

This folder contains the reusable Python package used by the experiments.

Main components:

- `dataset.py`  
  Builds paired consecutive half-orbit DEMETER sequences for LSTM input.

- `scaling.py`  
  Fits the scaler on nominal/non-seismic training sequences and applies the same transformation to train, validation, and test data.

- `models.py`  
  Defines the LSTM autoencoder architecture and related latent-space classifier modules.

- `training.py` and `training_modes.py`  
  Contain training routines for the LSTM autoencoder, including support for autoencoder-only and classifier-assisted modes.

- `anomaly_detection.py`  
  Computes reconstruction errors and detects aggregate or frequency-band anomalies.

- `SeismicCriteria.py`  
  Defines the spatial and temporal earthquake-association criterion.

- `Threshold_anomaly.py` and `Threshold_anomaly_WOG.py`  
  Perform seismic association analysis for detected anomalies.

- `timewindow_analysis.py`  
  Evaluates anomaly-earthquake ratios. Not for the current work.

- `fb_analyser.py`  
  Performs frequency-band-specific anomaly analysis. # Not used in the current analysis

- `classifier_evaluation.py` and `training_classifier.py`  
  Support optional latent-classifier experiments.

- `out_encoder.py`  
  Plots latent representations from the trained encoder.

Files with `_0` suffix are considered older backup versions and are not used as the active source code.

---

### `Final-Code/`

This folder contains runnable experiment scripts and notebooks.

It should include both:

```text
*.py
*.ipynb
```

 experiment files include:

- rolling-window LSTM-AE training;
- weight-updating mode experiments;
- weight-reinitialising mode experiments;
- trained-model reloading and result generation;
- anomaly-threshold analysis;
- seismic-association evaluation;
- plotting and result-summary scripts.
### `Final-Code/NOA`

This folder contains runnable experiment scripts and notebooks used for the SAO exploere ionogram data.

- SWARM folder contains the shared files for the swarm-ionogram # We havnt used it yet
- SAO_P, SAO, APP, _Fileds python files are the fuctions defnitions needed for reading the SAO files in the naming format "MHJ45_20050321(080)000000.SAO"
- Data Reader & pkl conversion -" SAO-Parsing_Data-Exampleipynb"
- Debugging sections - SAO-Parsing_Dataipynb {Use the data description file for better understanding}
- Staion_pkl-Correction.ipynb - Where two files of diff period of data of same station is combined
-SW22_48-NOA-Matching.ipynb - Spatial & Temporal Coincidences of the Demeter ANomalies & Ionosonde soundings



### `Models/`

This folder stores trained model files and training checkpoints.


```text
Model-Trained/
├── SW22_TW48/
│   ├── best_model_Hp_tw48-30dBG_A_w0.pth
│   ├── best_model_Hp_tw48-30dBG_A_w1.pth
│   └── ...
└── SW05_TW48/
    ├── best_model_Hp_tw48-30dBG_A_w0.pth
    ├── best_model_Hp_tw48-30dBG_In-A_w0.pth
    └── ...
```

Model naming convention:
`Each models tained is saved on folders of corresponding window criteria`
```text
best_model_Hp_tw{TW}-30dBG_{MODEL}_w{WINDOW}.pth
```

Example:

```text
best_model_Hp_tw48-30dBG_A_w0.pth
```

where:

- `tw48` means a 48-hour temporal window;
- `30dBG` means 30-day resampled background-corrected data;
- `A` means weight-updating mode;
- `In-A` means weight-reinitialising mode;
- `w0`, `w1`, ... indicate rolling-window index.

---

### `Data/`

This folder stores all data used in the study.

Large raw/intermediate data files are expected to be stored locally and should normally not be committed to GitLab unless the repository is private and storage rules allow it.

Main data files:

```text
Data/
├── storm_data.pkl
├── Down_Orbits-Q3-loc-Train-limitedgrids_RES.pkl
├── Main_earthquakes.csv
```

Expected contents:

- `storm_data.pkl`  
  Geomagnetic index data used for storm correction.

- `Down_Orbits-Q3-loc-Train-limitedgrids_RES.pkl`  
  Main DEMETER residual time-series dataframe containing location and frequency-band features (Q3 - `only consider Q3 and location - niothing else is needed`).

-  `EQ.csv`  
  Earthquake catalogue used for seismic association analysis.

---

### `Data/Bg_window_data/`

This folder stores rolling-window background-corrected DEMETER data.

Example files:

```text
Background_data-window_0.pkl
Background_data-window_1.pkl
Background_data-window_2.pkl
...
```

Each file corresponds to a rolling-window background correction and is used to extract:

- 12 months of training data;
- 3 months of validation data.

---

### `Data/Label_data/`

This folder stores the precomputed seismic/non-seismic sequence label files.

Example files:

```text
summary_df_train_30D-22SW-tw48_w0.csv
summary_df_val_30D-22SW-tw48_w0.csv
summary_df_train_30D-22SW-tw48_w1.csv
summary_df_val_30D-22SW-tw48_w1.csv
...
```
`Other than sw = 20, every other files have the above format with its spatial width mentioned`

These files label each paired half-orbit sequence as:

- `0`: non-seismic / nominal sequence;
- `1`: seismic-associated sequence under the selected spatial-temporal criterion.

The expected naming convention is:

```text
summary_df_{split}_30D-{SW}SW-tw{TW}_w{WINDOW}.csv
```

Example:

```text
SW =22
summary_df_train_30D-22SW-tw48_w0.csv
```

---
### `Data/RST-Label/`

This folder stores the precomputed seismic/non-seismic sequence label files applying the randomisation in time (this data can be used for the RST baseline calculation).

Example files:

```text
 RDM_df_train_30D-5SW-tw48_w0.csv
 RDM_df_train_30D-5SW-tw48_w1.csv
...
```



### `Result Files/` 

This folder contains the major output files produced during the study.

Expected contents include:

```text
Results/
├── SW22_TW48
  └── Retrain_result_Hp_B-tw48-30dBG_In-A.csv
  └── Blind
      └── Retrain_result_Hp_B1-tw48-30dBG_In-A.csv
```
Z-factor-files/
│       ├── *.csv

Provides the results of Z-factor analysis and average seismic anomaly ratio analysis fro all models individually and compined under each SW-TW Criteria

## Main workflow

The standard workflow is:

1. Load DEMETER residual Q3 data.
2. Load rolling-window background-corrected data.
3. Load earthquake catalogue.
4. Load or generate seismic/non-seismic sequence labels.
5. Build paired consecutive half-orbit sequences.
6. Fit scaler using only nominal/non-seismic training sequences.
7. Train or reload LSTM autoencoder models.
8. Compute reconstruction errors.
9. Apply percentile-based anomaly threshold.
10. Evaluate seismic association of anomalies.
11. Save anomaly indices, seismic indices, matched earthquake details, and final summary tables.

---

## Training modes

Two model-training modes are supported.

### A mode: weight updating

In this mode, each rolling window after the first starts from the previous window's best model.


Conceptually:

```text
w0: random initialization
w1: initialize from best model of w0
w2: initialize from best model of w1
...
```

This mode preserves learning continuity across rolling windows.

---

### In-A mode: weight reinitialising

In this mode, every rolling window starts from newly initialized weights.

Conceptually:

```text
w0: random initialization
w1: random initialization
w2: random initialization
...
```

This mode tests whether the detected association is reproducible without carrying weights from previous windows.

---

## Running with already trained models

If trained models already exist under:

```text
Models-Trained/SW22_TW48/
```

the experiment notebook or script can skip training and directly reload models.


## Important configuration example

A typical configuration for the selected SW22/TW48 experiment is:

```python
SW = 22
TW = 48


HIDDEN_SIZE = 8
NUM_LAYERS = 2
LATENT_DIM = 2
INPUT_SIZE = 11

PC = 98
```

---

## Output naming conventions

Final result CSV:

```text
Retrain_result_Hp_SW{SW}-tw{TW}-30dBG_{MODEL}.csv
```

Examples:

```text
Retrain_result_Hp_SW22-tw48-30dBG_A.csv
Retrain_result_Hp_SW22-tw48-30dBG_In-A.csv
```

Best model checkpoint:

```text
best_model_Hp_tw{TW}-30dBG_{MODEL}_w{WINDOW}.pth
```

Example:

```text
best_model_Hp_tw48-30dBG_A_w0.pth
```
---

## Reproducibility notes

For strict reproducibility, the following must match between training and result generation:

- DEMETER input dataframe;
- rolling-window background-corrected files;
- seismic label CSV files;
- train/validation date boundaries;
- feature columns;
- scaler fitting procedure;
- model architecture;
- hidden size;
- number of LSTM layers;
- latent dimension;
- anomaly threshold percentile;
- spatial window;
- temporal window;

The scaler is fitted only on nominal/non-seismic training data. If the label files or training data change, reconstruction errors and anomaly indices may also change.


## Example execution

Run the notebook:

```text
Experiments/DEMETER_Rolling_Training_A_InA_load_existing.ipynb
```

or execute the Python script:

``` In Cluster
Make an SH Source file file wiht requried GPU and memory specification and location of experiment file. 
example file - Retraining_Hp-B.sh
sbatch Retraining_Hp-B.sh
```

Before running, check the following paths in the configuration section:

```python
DEMETER_ROOT = Path to the folder
DATA_DIR = DEMETER_ROOT / "Data"
MODEL_DIR = DEMETER_ROOT / "Models-Trained"
RESULT_DIR = DEMETER_ROOT / "Result Files"
```

---

## Author

**Megha Babu**  
PhD Researcher
Research topic: data-driven deep-learning analysis of ionospheric electric-field perturbations and seismic association.
