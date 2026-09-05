"""
Retraining_TimesNet_HO-A.py

Converted from Retraining_Hp_HO-A.py:
- Replaces LSTMAutoencoder with TimesNetAutoencoder (optimized for T=17, F=11).
- Preserves 20 rolling windows (12m train / 3m val).
- Preserves weight-updating mode (_A): window i loads best_model from window i-1.
- Preserves normal-only training (label == 0).
- Preserves storm filtering (Dst, Kp, AE) and seismic criteria association analysis.
"""

import sys
import os
from datetime import datetime, timedelta
import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LSTM_PKG_PATH = os.path.join(os.path.dirname(__file__), "lstm")
for p in [PROJECT_ROOT, LSTM_PKG_PATH]:
    if p not in sys.path:
        sys.path.append(p)

# Import existing dataset, scaling, anomaly detector, and seismic evaluation classes
from lstm import HalfOrbitDataset
from lstm.scaling import scale_datasets_half_orbit
from lstm import AnomalyDetector_half_orbit
from lstm import SeismicCriteria_half_orbit
from lstm import SeismicAnalysis

plt.style.use('default')

# ==============================================================================
# REPRODUCIBILITY & DEVICE
# ==============================================================================
SEED = 201894
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on device: {DEVICE}")

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
sw = 4
tw = 48

# Model tag with TimesNet identifier
m = f'TimesNet-HO-sw{sw}_tw{tw}-30dBG_A'

# TimesNet Architecture parameters (T=17, F=11)
seq_len = 17
input_size = 11
d_model = 32      # Feature projection dimension
d_ff = 64         # Inception filter dimension
top_k = 2          # Top dominant frequencies for T=17
num_layers = 2    # Number of TimesBlocks

# Rolling-window parameters
train_months = 12
val_months = 3
stride = 3
min_data_points = 17

batch_size = 8
lr = 0.0001
num_epochs = 10
patience = 150
pc = 98           # Percentile threshold for reconstruction error

# ==============================================================================
# PATHS CONFIGURATION (Configured for local Windows system)
# ==============================================================================
BASE_DATA_DIR = os.environ.get("DEMETER_DATA_DIR", r"D:\GIT\Demeter-Anomaly-Detection-Framework\Data")
BG_WINDOW_DIR = os.path.join(BASE_DATA_DIR, "Bg_window_data")
output_dir1   = os.path.join(BG_WINDOW_DIR, "Halforbit")

output_model = os.path.join(PROJECT_ROOT, "models", "Model-Retraining", "Halforbit", f"SW{sw}10_TW{tw}")
os.makedirs(output_model, exist_ok=True)

output_dir = os.path.join(PROJECT_ROOT, "outputs", "TimesNet_BG", "Halforbit", f"SW{sw}10_TW{tw}")
os.makedirs(output_dir, exist_ok=True)

# Storm & Earthquake data
storm_data_path = os.path.join(BASE_DATA_DIR, "storm_data.pkl")
eq_path = os.path.join(BASE_DATA_DIR, "EQ.csv")

storm_data = pd.read_pickle(storm_data_path)
eq = pd.read_csv(eq_path, parse_dates=['Time'])
eq['Time'] = pd.to_datetime(eq['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
eq['Time'] = pd.to_datetime(eq['Time'])

# Storm thresholds
Dst = -50  # Remove if Dst < -50 nT
Kp = 3     # Remove if Kp > 3
AE = 500   # Remove if AE > 500 nT

storm_all = storm_data[
    (storm_data['Dst'] < Dst) |
    (storm_data['Kp']  > Kp)  |
    (storm_data['AE']  > AE)
].copy()
storm_all['Datetime'] = pd.to_datetime(storm_all['Datetime'])


# ==============================================================================
# TIMESNET MODEL DEFINITION (Self-Contained)
# ==============================================================================
class InceptionBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3):
        super(InceptionBlock2D, self).__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(2 * k + 1, 2 * k + 1), padding=k)
            for k in range(num_kernels)
        ])
        
    def forward(self, x):
        out = [conv(x) for conv in self.kernels]
        return torch.stack(out, dim=-1).mean(dim=-1)


class TimesBlock(nn.Module):
    def __init__(self, seq_len=17, d_model=32, d_ff=64, top_k=2, num_kernels=3):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.conv = nn.Sequential(
            InceptionBlock2D(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock2D(d_ff, d_model, num_kernels=num_kernels)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N = x.size()
        
        # 1D FFT along the temporal trajectory
        xf = torch.fft.rfft(x, dim=1)
        amplitudes = torch.abs(xf).mean(dim=0).mean(dim=-1)
        amplitudes[0] = 0  # Ignore DC
        
        k = min(self.k, amplitudes.size(0) - 1)
        _, top_indices = torch.topk(amplitudes, k)
        top_indices = top_indices.detach().cpu().numpy()
        
        periods = [max(2, int(np.round(T / idx))) for idx in top_indices if idx > 0]
        if not periods:
            periods = [2]
            
        period_weight = F.softmax(amplitudes[top_indices], dim=0)

        # 2D Folding & Inception Convolution
        res = []
        for i, p in enumerate(periods):
            length = (((T - 1) // p) + 1) * p
            padding = length - T
            x_padded = F.pad(x, (0, 0, 0, padding)) if padding > 0 else x

            x_2d = x_padded.reshape(B, length // p, p, N).permute(0, 3, 1, 2)
            conv_out = self.conv(x_2d)
            out_1d = conv_out.permute(0, 2, 3, 1).reshape(B, length, N)[:, :T, :]
            res.append(out_1d * period_weight[i])

        res = torch.stack(res, dim=-1).sum(dim=-1) + x
        return self.norm(res)


class TimesNetAutoencoder(nn.Module):
    """
    Direct replacement for LSTMAutoencoder:
    Input:  (B, 17, 11)
    Output: (B, 17, 11)
    """
    def __init__(self, seq_len=17, input_size=11, d_model=32, d_ff=64, num_layers=2, top_k=2):
        super(TimesNetAutoencoder, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len=seq_len, d_model=d_model, d_ff=d_ff, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.projection = nn.Linear(d_model, input_size)

    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.projection(h)


# ==============================================================================
# TRAINING FUNCTION FOR TIMESNET
# ==============================================================================
def train_timesnet_half_orbit(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=0.0001,
    patience=150,
    best_model_path="best_model.pth",
    loss_plot_path=None,
    device=DEVICE
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_batch_loss = []
        for batch in train_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)

            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_batch_loss.append(loss.item())

        avg_train_loss = np.mean(train_batch_loss)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_batch_loss = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(device)
                recon = model(x)
                val_loss = criterion(recon, x)
                val_batch_loss.append(val_loss.item())

        avg_val_loss = np.mean(val_batch_loss)
        val_losses.append(avg_val_loss)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{num_epochs:02d}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] Patience reached at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Plot loss curve
    if loss_plot_path:
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Train Loss", color="royalblue")
        plt.plot(val_losses, label="Val Loss", color="darkorange")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("TimesNet Training Curve")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=300)
        plt.close()

    return model, train_losses, val_losses


# ==============================================================================
# HELPER FUNCTIONS: PROCESSING SPLITS & SEISMIC ASSOCIATION
# ==============================================================================
def process_split(name, model, dataloader, num_features=11,
                  threshold_agg=None, threshold_fb=None, threshold_percentile=None, plot=False):
    detector = AnomalyDetector_half_orbit(model, dataloader=dataloader, num_features=num_features)

    errors_agg = detector.compute_reconstruction_errors_agg(dataloader)
    errors_fb = detector.compute_reconstruction_errors_fb(dataloader)

    if threshold_agg is None:
        threshold_agg = np.percentile(errors_agg, threshold_percentile)
    if threshold_fb is None:
        threshold_fb = np.percentile(errors_fb, threshold_percentile, axis=0)

    anomalies_agg = detector.detect_anomalies_agg(errors_agg, threshold_agg)
    anomalies_fb = detector.detect_anomalies_fb(errors_fb, threshold_fb)

    return {
        "detector": detector,
        "errors_agg": errors_agg,
        "errors_fb": errors_fb,
        "threshold_agg": threshold_agg,
        "threshold_fb": threshold_fb,
        "anomalies_agg": anomalies_agg,
        "anomalies_fb": anomalies_fb
    }


def correct_anomalies_for_storms(datetime_sequences, anomalies_agg, anomalies_fb, storm_all):
    storm_data_indices = []
    storm_all['Datetime'] = pd.to_datetime(storm_all['Datetime'])

    for idx, seq in enumerate(datetime_sequences):
        seq_times = pd.to_datetime(seq)
        start_t, end_t = seq_times[0], seq_times[-1]
        if storm_all['Datetime'].between(start_t, end_t).any():
            storm_data_indices.append(idx)

    corrected_anomalies_agg = [
        idx for idx in anomalies_agg if idx not in storm_data_indices
    ]

    corrected_anomalies_fb = [[] for _ in range(len(anomalies_fb))]
    for i, anomalies in enumerate(anomalies_fb):
        for j in anomalies:
            if j not in storm_data_indices:
                corrected_anomalies_fb[i].append(j)

    return corrected_anomalies_agg, corrected_anomalies_fb, storm_data_indices


def run_seismic_analysis(
    dataset,
    test_dataset,
    eq,
    seismic_criteria,
    corrected_test_fb_anomalies,
    corrected_test_anomalies,
    model_name,
    output_dir,
    data_label=""
):
    sa = SeismicAnalysis(
        dataset=dataset,
        earthquake_catalog=eq,
        create_half_orbit_sequences=test_dataset.create_half_orbit_sequences,
        is_eq_fn=seismic_criteria.is_eq,
        model_name=f'{model_name}',
        threshold_label="98",
        mean_rst=0,
        sigma_rst=0,
        output_dir=output_dir
    )

    seismic_seqs_agg, matched_eqs_agg, missed_eqs_agg = sa.agg_analysis(
        data_label=data_label,
        anomalous_indices=corrected_test_anomalies,
        plot=False
    )

    total_sequences_agg = len(corrected_test_anomalies)
    anomalies_agg = len(seismic_seqs_agg)
    total_eq = len(matched_eqs_agg)

    if total_sequences_agg > 0:
        if anomalies_agg == total_sequences_agg:
            p_agg = (anomalies_agg) / total_sequences_agg
            p_agg1 = (anomalies_agg - 1) / total_sequences_agg
            agg_value = p_agg * 100
            agg_error = np.sqrt(p_agg1 * (1 - p_agg1) / total_sequences_agg) * 100
        elif anomalies_agg == 0:
            p_agg = (anomalies_agg) / total_sequences_agg
            p_agg1 = 1 / total_sequences_agg
            agg_value = p_agg * 100
            agg_error = np.sqrt(p_agg1 * (1 - p_agg1) / total_sequences_agg) * 100
        else:
            p_agg = anomalies_agg / total_sequences_agg
            agg_value = p_agg * 100
            agg_error = np.sqrt(p_agg * (1 - p_agg) / total_sequences_agg) * 100
    else:
        agg_value, agg_error = 0, 0

    return {
        "agg": {"value": agg_value, "error": agg_error, "total_eq": total_eq, "seismic_indices": seismic_seqs_agg}
    }


def generate_windows(start_date, end_date, train_months, val_months):
    windows = []
    current = pd.to_datetime(start_date)

    while current + pd.DateOffset(months=train_months + val_months) < pd.to_datetime(end_date):
        train_start = current
        train_end   = current + pd.DateOffset(months=train_months)
        val_end     = train_end + pd.DateOffset(months=val_months)

        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": val_end
        })

        current = current + pd.DateOffset(months=stride)

    return windows


def plot_error_distributions(i, m, train, val, pc=pc, output_dir=None, save=False, n_bins=40):
    errors_train = pd.Series(train['errors_agg']).dropna().to_numpy()
    errors_val   = pd.Series(val['errors_agg']).dropna().to_numpy()

    if len(errors_train) == 0 or len(errors_val) == 0:
        return

    threshold_train = np.percentile(errors_train, pc)
    threshold_val   = np.percentile(errors_val, pc)

    all_errors = np.concatenate([errors_train, errors_val])
    min_err, max_err = all_errors.min(), all_errors.max()
    bins = np.linspace(min_err, max_err, n_bins + 1)

    plt.figure(figsize=(12, 6))
    plt.hist(errors_train, bins=bins, color='tab:blue', alpha=0.6, label=f'Train_w{i}', edgecolor='black')
    plt.hist(errors_val, bins=bins, color='tab:orange', alpha=0.6, label=f'Validation_w{i}', edgecolor='black')

    plt.axvline(threshold_train, color='tab:blue', linestyle='--', linewidth=2, label=f'Train {pc}th pct ({threshold_train:.3f})')
    plt.axvline(threshold_val, color='tab:orange', linestyle='--', linewidth=2, label=f'Val {pc}th pct ({threshold_val:.3f})')

    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'TimesNet Reconstruction Error Distribution (Window {i})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    if save and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{output_dir}/Hist-Errordistribution-pc{pc}-TimesNet_m{m}_w{i}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ==============================================================================
# MAIN ROLLING-WINDOW TRAINING & EVALUATION LOOP
# ==============================================================================
windows_train = generate_windows(
    start_date="2005-01-01 00:00:00",
    end_date="2010-01-02 00:00:00",
    train_months=train_months,
    val_months=val_months
)

all_results = []

print(f"\n[*] Starting TimesNet Rolling Window Retraining ({len(windows_train)} windows)...")

for i, w in enumerate(windows_train):
    tag_l = f'tw{tw}_w{i}'
    tag = f'Hp_{m}_w{i}'

    print(f"\n{'='*90}")
    print(f"Window {i}: Train {w['train_start']} -> {w['train_end']} | Val {w['val_start']} -> {w['val_end']}")
    print(f"{'='*90}")

    # Load background-corrected data for this window
    bg_file = os.path.join(BG_WINDOW_DIR, f"Background_data-window_{i}.pkl")
    if not os.path.exists(bg_file):
        bg_file = f"/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Bg_window_data/Background_data-window_{i}.pkl"
    dfw = pd.read_pickle(bg_file)
    dfw = dfw.loc[:, ~dfw.columns.str.startswith('Q3')]

    train_set = dfw[(dfw.index >= w['train_start']) & (dfw.index < w['train_end'])]
    val_set   = dfw[(dfw.index >= w['val_start'])   & (dfw.index < w['val_end'])]
    test_set  = pd.DataFrame(dfw[dfw.index >= '2005-10-01'])

    # Build datasets
    train_dataset = HalfOrbitDataset(train_set, min_data_points=min_data_points)
    val_dataset   = HalfOrbitDataset(val_set,   min_data_points=min_data_points)
    test_dataset  = HalfOrbitDataset(test_set,  min_data_points=min_data_points)

    seismic_criteria = SeismicCriteria_half_orbit(spatial_width_a=sw, spatial_width_b=10, time_window_hours=tw)

    # Load seismic labels for normal-only training
    train_label_csv = os.path.join(output_dir1, f"summary_df_train_30D-{sw}_10SW-{tag_l}.csv")
    val_label_csv   = os.path.join(output_dir1, f"summary_df_val_30D-{sw}_10SW-{tag_l}.csv")

    df_train_labels = pd.read_csv(train_label_csv)
    df_val_labels   = pd.read_csv(val_label_csv)

    # Filter to label == 0 (non-seismic) for normal model training
    train_dataset_normal = HalfOrbitDataset(train_set, df_train_labels, min_data_points, use_label_0_only=True)
    val_dataset_normal   = HalfOrbitDataset(val_set,   df_val_labels,   min_data_points, use_label_0_only=True)

    # Scale datasets (fit on normal training orbits, transform val/test)
    scaled_train_data_n, scaled_val_data_n, scaled_test_data, mean = scale_datasets_half_orbit(
        train_dataset_normal,
        train_dataset,
        val_dataset_normal,
        test_dataset,
        fit=True
    )

    train_data_loader_n = DataLoader(scaled_train_data_n, batch_size=batch_size, shuffle=True)
    val_data_loader_n   = DataLoader(scaled_val_data_n,   batch_size=batch_size, shuffle=False)

    print(f"Length train_loader (normal only): {len(train_data_loader_n)} | val_loader: {len(val_data_loader_n)}")

    # Initialize TimesNet model
    model = TimesNetAutoencoder(
        seq_len=seq_len,
        input_size=input_size,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        top_k=top_k
    ).to(DEVICE)

    best_model_path = os.path.join(output_model, f"best_model_{tag}.pth")
    loss_plot_path  = os.path.join(output_model, f"loss_curve_model_{tag}.png")

    # Weight-Updating Mode (_A): Window 0 trains from scratch, Window i > 0 warm-starts from Window i-1
    if i == 0:
        print("[*] Window 0: Training fresh TimesNet model...")
    else:
        try:
            prev_tag = f'Hp_{m}_w{i-1}'
            prev_best = os.path.join(output_model, f"best_model_{prev_tag}.pth")
            model.load_state_dict(torch.load(prev_best, map_location=DEVICE))
            print(f"[Warm start] Successfully loaded previous weights from: {prev_best}")
        except Exception as e:
            print(f"[Warm start skipped] {e}")

    model, train_losses, val_losses = train_timesnet_half_orbit(
        model=model,
        train_loader=train_data_loader_n,
        val_loader=val_data_loader_n,
        num_epochs=num_epochs,
        lr=lr,
        patience=patience,
        best_model_path=best_model_path,
        loss_plot_path=loss_plot_path,
        device=DEVICE
    )

    print(f"[OK] Saved best model -> {best_model_path}")

    # Scale full data (unfiltered) for anomaly detection and evaluation
    scaled_train_data, scaled_val_data, _, _ = scale_datasets_half_orbit(
        train_dataset_normal,
        train_dataset,
        val_dataset,
        test_dataset,
        fit=False
    )

    train_data_loader = DataLoader(scaled_train_data, batch_size=1, shuffle=False)
    val_data_loader   = DataLoader(scaled_val_data,   batch_size=1, shuffle=False)

    train = process_split(f"Train_{tag_l}", model, train_data_loader, threshold_percentile=pc, num_features=11, plot=False)
    val   = process_split(f"Validation_{tag_l}", model, val_data_loader, num_features=11,
                          threshold_agg=train["threshold_agg"],
                          threshold_fb=train["threshold_fb"], plot=False)

    plot_error_distributions(i, m, train, val, pc=pc, output_dir=output_dir, save=True)

    val_sequences, val_datetime_sequences, val_lat_long_sequences, _ = val_dataset.create_half_orbit_sequences(val_set)
    train_sequences, train_datetime_sequences, train_lat_long_sequences, _ = train_dataset.create_half_orbit_sequences(train_set)

    # Storm Correction
    corrected_val_anomalies_agg, corrected_val_anomalies_fb, storm_val_data_indices = correct_anomalies_for_storms(
        val_datetime_sequences, val['anomalies_agg'], val['anomalies_fb'], storm_all
    )
    corrected_train_anomalies_agg, corrected_train_anomalies_fb, storm_train_data_indices = correct_anomalies_for_storms(
        train_datetime_sequences, train['anomalies_agg'], train['anomalies_fb'], storm_all
    )

    val_anomalies_before = len(val['anomalies_agg'])
    val_anomalies_after  = len(corrected_val_anomalies_agg)
    train_anomalies_before = len(train['anomalies_agg'])
    train_anomalies_after  = len(corrected_train_anomalies_agg)

    # Seismic Association Analysis
    results_val_sc = run_seismic_analysis(
        val_set, val_dataset, eq, seismic_criteria,
        corrected_val_anomalies_fb, corrected_val_anomalies_agg,
        model_name=f"stormC_EQ_{tag}", output_dir=output_dir, data_label=f"Val_{tag_l}"
    )

    results_train_sc = run_seismic_analysis(
        train_set, train_dataset, eq, seismic_criteria,
        corrected_train_anomalies_fb, corrected_train_anomalies_agg,
        model_name=f"stormC_EQ_{tag}", output_dir=output_dir, data_label=f"Train_{tag_l}"
    )

    results_val = run_seismic_analysis(
        val_set, val_dataset, eq, seismic_criteria,
        val['anomalies_fb'], val['anomalies_agg'],
        model_name=f"EQ_{tag}", output_dir=output_dir, data_label=f"Val_{tag_l}"
    )

    results_train = run_seismic_analysis(
        train_set, train_dataset, eq, seismic_criteria,
        train['anomalies_fb'], train['anomalies_agg'],
        model_name=f"EQ_{tag}", output_dir=output_dir, data_label=f"Train_{tag_l}"
    )

    all_results.append({
        "split": "Val",
        "window": i,
        "anomalies": val_anomalies_before,
        "anomalies_sc": val_anomalies_after,
        "agg_value": results_val["agg"]["value"],
        "agg_error": results_val["agg"]["error"],
        "agg_value_sc": results_val_sc["agg"]["value"],
        "agg_error_sc": results_val_sc["agg"]["error"],
        "agg_total_eq": results_val["agg"]["total_eq"],
        "agg_total_eq_sc": results_val_sc["agg"]["total_eq"],
        "seismic_indices_sc": results_val_sc["agg"]["seismic_indices"], 
        "seismic_indices": results_val["agg"]["seismic_indices"], 
    })

    all_results.append({
        "split": "Train",
        "window": i,
        "anomalies": train_anomalies_before,
        "anomalies_sc": train_anomalies_after,
        "agg_value": results_train["agg"]["value"],
        "agg_error": results_train["agg"]["error"],
        "agg_value_sc": results_train_sc["agg"]["value"],
        "agg_error_sc": results_train_sc["agg"]["error"], 
        "agg_total_eq": results_train["agg"]["total_eq"],
        "agg_total_eq_sc": results_train_sc["agg"]["total_eq"],
        "seismic_indices_sc": results_train_sc["agg"]["seismic_indices"], 
        "seismic_indices": results_train["agg"]["seismic_indices"], 
    })

    print(pd.DataFrame(all_results).tail(2))

# Save full results to CSV
df_results = pd.DataFrame(all_results)
csv_path = os.path.join(output_dir, f"Retrain_result_Hp-{m}.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n[DONE] Full TimesNet results saved to: {csv_path}")
