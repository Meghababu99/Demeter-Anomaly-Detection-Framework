# %%

import sys
sys.path.append('/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter') 

from lstm import HalfOrbitDataset
from lstm.scaling import scale_datasets_half_orbit
from lstm import LSTMAutoencoder
from lstm import plot_encoded_representation
from lstm import AnomalyDetector_half_orbit
from lstm import SeismicAnomalyAnalyzer
from lstm import SeismicCriteria_half_orbit
from lstm import train_lstm_ae_mode_half_orbit
from lstm import SeismicAnalysis
from lstm import SeismicRatioEvaluator

from datetime import datetime

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
from shapely.geometry import Point, Polygon
from datetime import timedelta
import torch.optim as optim
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
from datetime import timedelta
from shapely.geometry import Polygon, Point
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import numpy as np

# kasdfkhaflkj
plt.style.use('default')

torch.manual_seed(201894)
np.random.seed(201894)

sw =4
tw =48

m=f'HO-sw{sw}_tw{tw}-30dBG_A'
hidden_size = 8






output_dir1 = f"/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Bg_window_data/Halforbit/"
os.makedirs(output_dir1, exist_ok=True)
output_model  =f"/storage3/DSIP/Demeter/Demeter/models/Model-Retraining/Halforbit/SW{sw}10_TW{tw}"
os.makedirs(output_model, exist_ok=True)

output_dir = f"/storage3/DSIP/Demeter/Demeter/outputs/New_BG/Halforbit/SW{sw}10_TW{tw}"
os.makedirs(output_dir, exist_ok=True)


# parameters
train_months = 12
val_months   = 3
min_data_points = 17
stride = 3
latent_dim = 2

batch_size = 8
lr = 0.0001
num_layers = 2
num_epochs = 10

patience= 150

pc =98



# # %%
storm_data = pd.read_pickle('/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/storm_data.pkl') 

eq = pd.read_csv("/home/mbabu/EQ_Data/EQ.csv", parse_dates=['Time'])                     
# eq = pd.read_csv("/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Main_earthquakes.csv", parse_dates=['Time'])
eq['Time'] = pd.to_datetime(eq['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
eq['Time'] = pd.to_datetime(eq['Time'])


Dst = -50 # if dst < -50 nT it has to be removed
Kp = 3 # if kp > 3 nT it has to be removed
AE = 500 # if AE >500nT it has to be removed 

storm_all = storm_data[
    (storm_data['Dst'] < Dst) |
    (storm_data['Kp']  > Kp) |
    (storm_data['AE']  > AE)
].copy()


storm_all['Datetime'] = pd.to_datetime(storm_all['Datetime'])



Dst = -50 # if dst < -50 nT it has to be removed
Kp = 3 # if kp > 3 nT it has to be removed
AE = 500 # if AE >500nT it has to be removed 

storm_all = storm_data[
    (storm_data['Dst'] < Dst) |
    (storm_data['Kp']  > Kp) |
    (storm_data['AE']  > AE)
].copy()


storm_all['Datetime'] = pd.to_datetime(storm_all['Datetime'])



# %%

def process_split(name, model, dataloader, num_features,
                  threshold_agg=None, threshold_fb=None, threshold_percentile=None,plot=None):
    detector = AnomalyDetector_half_orbit(model, dataloader=dataloader, num_features=num_features)

    errors_agg = detector.compute_reconstruction_errors_agg(dataloader)
    errors_fb = detector.compute_reconstruction_errors_fb(dataloader)

    if threshold_agg is None:
        threshold_agg = np.percentile(errors_agg, threshold_percentile)
    if threshold_fb is None:
        threshold_fb = np.percentile(errors_fb, threshold_percentile, axis=0)

    anomalies_agg = detector.detect_anomalies_agg(errors_agg, threshold_agg)
    anomalies_fb = detector.detect_anomalies_fb(errors_fb, threshold_fb)
    if plot:
        detector.plot_feature_errors(errors_fb,title=f"{name} Data Error Distribution ",threshold=threshold_fb,percentile=threshold_percentile, save_path=f"{output_dir}\FB_Error-{name}_model{m}.png")
        detector.plot_reconstruction_error(errors_agg,title=f"{name} Data Error Distribution ",threshold=threshold_agg,percentile=threshold_percentile, save_path=f"{output_dir}\Agg_Error-{name}_model{m}.png")
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
    """
    Removes storm-affected sequences from anomalies.

    Parameters:
    - datetime_sequences: list of list-like sequences of timestamps
    - anomalies_agg: list of indices (int) of aggregated anomalies
    - anomalies_fb: list of lists of per-sequence anomaly indices
    - storm_all: DataFrame with 'Datetime' column containing storm event timestamps

    Returns:
    - corrected_anomalies_agg: filtered list of anomalies_agg
    - corrected_anomalies_fb: filtered list of lists of anomalies_fb
    - storm_data_indices: indices of sequences affected by storms
    """
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



# %%



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
  

    # Create instance
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

    # --- Aggregate analysis ---
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
            p_agg1= (anomalies_agg-1) / total_sequences_agg
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

    results = {
        "agg": {"value": agg_value, "error": agg_error, "total_eq": total_eq,"seismic_indices": seismic_seqs_agg }
    }

    return results




# function to generate rolling windows
def generate_windows(start_date, end_date, train_months, val_months):
    """
    Generate train/val windows from start to end date.
    Each window: 12m train + 3m val.
    """
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
    """
    Plot count-based histograms (non-density) of reconstruction errors
    for train and validation datasets, with separate percentile thresholds.

    Parameters
    ----------
    i : int or str
        Window/fold index for labeling.
    m : int or str
        Model/config identifier for filename.
    train, val : pd.DataFrame
        DataFrames with an 'errors_agg' column of reconstruction errors.
    pc : int, default=98
        Percentile threshold to plot for each dataset.
    output_dir : str, optional
        Directory to save the figure if save=True.
    save : bool, default=False
        Whether to save the plot to disk.
    n_bins : int, default=40
        Number of bins in the histogram.
    """

    # --- Extract error arrays ---
    errors_train = pd.Series(train['errors_agg']).dropna().to_numpy()
    errors_val   = pd.Series(val['errors_agg']).dropna().to_numpy()

    if len(errors_train) == 0 or len(errors_val) == 0:
        raise ValueError("Empty error arrays found — check that both train and val have 'errors_agg' values.")

    # --- Compute separate thresholds ---
    threshold_train = np.percentile(errors_train, pc)
    threshold_val   = np.percentile(errors_val, pc)

    # --- Shared bin edges based on combined min/max ---
    all_errors = np.concatenate([errors_train, errors_val])
    min_err, max_err = all_errors.min(), all_errors.max()
    bins = np.linspace(min_err, max_err, n_bins + 1)

    # --- Plot ---
    plt.figure(figsize=(12, 6))

    # Histograms (counts, not density)
    plt.hist(errors_train, bins=bins, color='tab:blue', alpha=0.6,
             label=f'Train_w{i}', edgecolor='black')
    plt.hist(errors_val, bins=bins, color='tab:orange', alpha=0.6,
             label=f'Validation_w{i}', edgecolor='black')

    # --- Threshold lines ---
    plt.axvline(threshold_train, color='tab:blue', linestyle='--', linewidth=2,
                label=f'Train {pc}th pct ({threshold_train:.3f})')
    plt.axvline(threshold_val, color='tab:orange', linestyle='--', linewidth=2,
                label=f'Val {pc}th pct ({threshold_val:.3f})')

    # --- Annotate threshold values ---
    ylim = plt.ylim()
    plt.text(threshold_train, ylim[1]*0.9, f'{threshold_train:.3f}',
             rotation=90, color='tab:blue', va='bottom', ha='right', fontsize=10, fontweight='bold')
    plt.text(threshold_val, ylim[1]*0.8, f'{threshold_val:.3f}',
             rotation=90, color='tab:orange', va='bottom', ha='right', fontsize=10, fontweight='bold')

    # --- Style ---
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Reconstruction Error Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # --- Save or show ---
    if save and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{output_dir}/Hist-Errordistribution-pc{pc}-HP_m{m}_w{i}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved histogram → {fname}")
    else:
        plt.show()



# %%

        
# Example: generate windows
windows_train = generate_windows(start_date="2005-01-01 00:00:00", end_date="2010-01-02 00:00:00",
                           train_months=train_months, val_months=val_months)
all_results = []
# iterate through windows
for i, w in enumerate(windows_train):
    tag_l = f'tw{tw}_w{i}'
    tag= f'Hp_{m}_w{i}'

    print(f"Window {i}: Train {w['train_start']} → {w['train_end']}, "
          f"Val {w['val_start']} → {w['val_end']}")

    # subset data
    dfw = pd.read_pickle(f"/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Bg_window_data/Background_data-window_{i}.pkl")

    # print(dfw)
    dfw = dfw.loc[:, ~dfw.columns.str.startswith('Q3')]
    train_set = dfw[(dfw.index >= w['train_start']) & (dfw.index < w['train_end'])]
    val_set  = dfw[(dfw.index >= w['val_start'])   & (dfw.index < w['val_end'])]
    test_set  = pd.DataFrame(dfw[dfw.index >= '2005-10-01'])


    # datasets (non-seismic only for fitting scaler + training)
    train_dataset = HalfOrbitDataset(train_set, min_data_points=min_data_points)
    val_dataset   = HalfOrbitDataset(val_set,   min_data_points=min_data_points)
    test_dataset   = HalfOrbitDataset(test_set,   min_data_points=min_data_points)

    seismic_criteria = SeismicCriteria_half_orbit(spatial_width_a=sw,spatial_width_b=10, time_window_hours=tw)
    
    df_train_labels =pd.read_csv(os.path.join(output_dir1, f"summary_df_train_30D-{sw}_10SW-{tag_l}.csv"))
    df_val_labels   =pd.read_csv(os.path.join(output_dir1,f"summary_df_val_30D-{sw}_10SW-{tag_l}.csv"))
    # Build datasets that only keep pairs with label==0 (non-seismic) for normal training
    # HalfOrbitPairDataset will filter when you pass df_labels and use_label_0_only=True
    train_dataset_normal = HalfOrbitDataset(train_set, df_train_labels, min_data_points, use_label_0_only=True)
 
    val_dataset_normal   = HalfOrbitDataset(val_set,   df_val_labels,   min_data_points, use_label_0_only=True)

    # Scale datasets (fit on train normals, transform val/test)
    scaled_train_data_n, scaled_val_data_n, scaled_test_data, mean = scale_datasets_half_orbit(
        train_dataset_normal,     # fit scaler on this
        train_dataset,       # unfiltered train base (for consistency with your API)
        val_dataset_normal,
        test_dataset,
        fit=True
    )
    # DataLoaders
    train_data_loader_n = DataLoader(scaled_train_data_n, batch_size=batch_size,shuffle=True)
    val_data_loader_n = DataLoader(scaled_val_data_n, batch_size=batch_size,shuffle=False)
    # blind_data_loader = DataLoader(scaled_blind_data, batch_size=batch_size,shuffle=False)
    
    print("TRAINING DATA INFORMATION")
    print("length of train_loader:", len(train_data_loader_n))
    print("length of val_loader:",   len(val_data_loader_n))
    print("length of train_data:",   len(scaled_train_data_n))
    print("length of val_data:",     len(scaled_val_data_n))


    # # Build model
    model = LSTMAutoencoder(
        input_size=11,
        hidden_size=hidden_size,
        num_layers= num_layers,
        latent_dim=latent_dim
    )
    # Paths per time window
    all_epochs_dir = os.path.join(output_model, f"check_{tag}")
    best_model_path = os.path.join(output_model, f"best_model_{tag}.pth")
    loss_plot_path  = os.path.join(output_model, f"loss_curve_model_{tag}.png")
    os.makedirs(all_epochs_dir, exist_ok=True)

    # Train & save
    if i ==0:
        model, train_losses, val_losses = train_lstm_ae_mode_half_orbit(
            model,
            train_data_loader_n,
            val_data_loader_n,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            early_stopping=True,
            save_all_epochs=True,
            all_epochs_dir=all_epochs_dir,
            best_model_path=best_model_path,
            plot_loss=True,
            loss_plot_path=loss_plot_path,
            input_size=11,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            seq=17,
            mode='autoencoder'
        )
  
    else:
        try:
            prev_tag = f'Hp_{m}_w{i-1}'
            prev_best = os.path.join(output_model, f"best_model_{prev_tag}.pth")
            model.load_state_dict(torch.load(prev_best, map_location=torch.device("cpu")))
            print(f"[Warm start] Loaded {prev_best}")
        except Exception as e:
            print(f"[Warm start skipped] {e}")
        model, train_losses, val_losses = train_lstm_ae_mode_half_orbit(
            model,
            train_data_loader_n,
            val_data_loader_n,
            num_epochs=num_epochs,
            lr=lr,
            patience=patience,
            early_stopping=True,
            save_all_epochs=True,
            all_epochs_dir=all_epochs_dir,
            best_model_path=best_model_path,
            plot_loss=True,
            loss_plot_path=loss_plot_path,
            input_size=11,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            seq=17,
            mode='autoencoder'
        )
        


    print(f"[OK] Saved best model -> {best_model_path}")
    print(f"[OK] Saved loss curve -> {loss_plot_path}")

    scaled_train_data, scaled_val_data, _,_ = scale_datasets_half_orbit(
        train_dataset_normal,     # fit scaler on this
        train_dataset,       # unfiltered train base (for consistency with your API)
        val_dataset,
        test_dataset,
        fit=False
    )


    # DataLoaders
    train_data_loader = DataLoader(scaled_train_data, batch_size=1,shuffle=False)
    val_data_loader = DataLoader(scaled_val_data, batch_size=1,shuffle=False)
    # for 
    print("Results DATA INFORMATION")
    print("length of train_loader:", len(train_data_loader))
    print("length of val_loader:",   len(val_data_loader))
    print("length of train_data:",   len(scaled_train_data))
    print("length of val_data:",     len(scaled_val_data))

    train = process_split(f"Train_{tag_l}", model, train_data_loader, threshold_percentile=pc,num_features=11,plot=False)


    val = process_split(f"Validation_{tag_l}", model, val_data_loader,num_features=11,
                        threshold_agg=train["threshold_agg"],
                        threshold_fb=train["threshold_fb"],plot=False)
    

    plot_error_distributions(i, m, train, val, pc=pc, output_dir=output_dir, save=True)

    val_sequences, val_datetime_sequences, val_lat_long_sequences, _ = val_dataset.create_half_orbit_sequences(val_set)
    
    train_sequences, train_datetime_sequences, train_lat_long_sequences, _ = train_dataset.create_half_orbit_sequences(train_set)


    corrected_val_anomalies_agg, corrected_val_anomalies_fb, storm_val_data_indices = correct_anomalies_for_storms(val_datetime_sequences, val['anomalies_agg'], val['anomalies_fb'], storm_all)

    corrected_train_anomalies_agg, corrected_train_anomalies_fb, storm_train_data_indices = correct_anomalies_for_storms(train_datetime_sequences,train['anomalies_agg'],train['anomalies_fb'],storm_all)

    val_anomalies_before = len(val['anomalies_agg'])
    val_anomalies_after = len(corrected_val_anomalies_agg)


    train_anomalies_before = len(train['anomalies_agg'])
    train_anomalies_after = len(corrected_train_anomalies_agg)


    results_val_sc = run_seismic_analysis(
    val_set,
    val_dataset,
    eq,
    seismic_criteria,
    corrected_val_anomalies_fb,
    corrected_val_anomalies_agg,
    model_name =f"stormC_EQ_{tag}" ,
    output_dir=output_dir,
    data_label=f"Val_{tag_l}")


    results_train_sc = run_seismic_analysis(
    train_set,
    train_dataset,
    eq,
    seismic_criteria,
    corrected_train_anomalies_fb,
    corrected_train_anomalies_agg,
    model_name =f"stormC_EQ_{tag}" ,
    output_dir=output_dir,
    data_label=f"Train_{tag_l}")
    
    results_val= run_seismic_analysis(
    val_set,
    val_dataset,
    eq,
    seismic_criteria,
    val['anomalies_fb'],
    val['anomalies_agg'],
    model_name =f"EQ_{tag}" ,
    output_dir=output_dir,
    data_label=f"Val_{tag_l}")


    results_train = run_seismic_analysis(
    train_set,
    train_dataset,
    eq,
    seismic_criteria,
    train['anomalies_fb'],
    train['anomalies_agg'],
    model_name =f"EQ_{tag}" ,
    output_dir=output_dir,
    data_label=f"Train_{tag_l}")
    # Aggregate
 

    all_results.append({
        "split": "Val",
        "anomalies": val_anomalies_before,
        "anomalies_sc": val_anomalies_after,
        "agg_value": results_val["agg"]["value"],
        "agg_error": results_val["agg"]["error"],
        "agg_value_sc": results_val_sc["agg"]["value"],
        "agg_error_sc": results_val_sc["agg"]["error"],
        "agg_total_eq": results_val["agg"]["total_eq"],
        "agg_total_eq_sc": results_val_sc["agg"]["total_eq"],
        "seismic_indices_sc":results_val_sc["agg"]["seismic_indices"], 
        "seismic_indices":results_val["agg"]["seismic_indices"], 
    })
 

    all_results.append({
        "split": "Train",
        "anomalies": train_anomalies_before,
        "anomalies_sc": train_anomalies_after,
        "agg_value": results_train["agg"]["value"],
        "agg_error": results_train["agg"]["error"],
        "agg_value_sc": results_train_sc["agg"]["value"],
        "agg_error_sc": results_train_sc["agg"]["error"], 
        "agg_total_eq": results_train["agg"]["total_eq"],
        "agg_total_eq_sc": results_train_sc["agg"]["total_eq"],
        "seismic_indices_sc":results_train_sc["agg"]["seismic_indices"], 
        "seismic_indices":results_train["agg"]["seismic_indices"], 
    })

    print(pd.DataFrame(all_results))

df_results = pd.DataFrame(all_results)

# Save to CSV
csv_path = os.path.join(output_dir, f"Retrain_result_Hp-{m}.csv")
df_results.to_csv(csv_path, index=False)

print(f"Results saved to {csv_path}")
