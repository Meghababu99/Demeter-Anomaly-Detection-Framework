# %%

# FOR THE PREPERATION OF ROLLING WINDOW DATA.
# THE BACKGROUND CORRECTION PERFORMED USING THE 30D RESAMPLING WINDOWS
# ALL DATA SAVED IN THE NAME OF Background_data-window_{i}.pkl , WHere i represents the window number 0-19





#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import os
from scipy import stats



from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
plt.style.use('default')

# torch.manual_seed(201894)
np.random.seed(201894)

# %%
data = pd.read_pickle("/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Down_Orbits-max-val-location.pkl") #GRID bsed data 


# %%
output_dir = r"/home/mbabu/GRID-METHODS/LSTM-AE-HO/Demeter/Data/Bg_window_data"
os.makedirs(output_dir, exist_ok=True)


# %%
import pandas as pd


# function to generate rolling windows
def generate_windows(start_date, end_date, train_months=12, val_months=3,stride =3):
    """
    Generate train/val windows from start to end date.
    Each window: 12m train + 3m val.
    """
    """
    Create sliding windows over a date range.

    Each window has:
      - train: [train_start, train_end)   (12 months by default)
      - val:   [train_end,   val_end)     (3 months by default)

    Windows advance by `stride_months` (3 by default).
    Only windows fully contained in [start_date, end_date] are returned.

    Parameters
    ----------
    start_date : str | pd.Timestamp
        Inclusive start of the timeline to consider.
    end_date : str | pd.Timestamp
        Exclusive end of the timeline to consider.
    train_months : int
        Number of months in the training segment.
    val_months : int
        Number of months in the validation segment.
    stride_months : int
        Step size (in months) to slide the window forward.

    Returns
    -------
    list[dict]
        Each dict contains:
        {
          "train_start": Timestamp,
          "train_end":   Timestamp,
          "val_start":   Timestamp,
          "val_end":     Timestamp,
        }

    Notes
    -----
    - Train is half-open: index >= train_start & index < train_end
    - Val is half-open:   index >= val_start   & index < val_end
    - Use these directly in boolean masks for slicing your DataFrame.
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

        # slide window forward (e.g. by 3 months, or 1 month depending on design)
        current = current + pd.DateOffset(months=stride)

    return windows




# %%
windows_train = generate_windows(
    start_date="2005-01-01 00:00:00",
    end_date="2011-01-02 00:00:00",
    train_months=12,
    val_months=3           # or "Europe/Rome" if your index is tz-aware
)



# %%
# Define the model function
def model_func(t, a, b, c, d, e):
    return a + t * b + c * np.sin((2 * np.pi * t) / d + e)

# Define the function to calculate resampled stats
def calculate_filtered_stats(data, column):
    mean_value = data[column].mean()
    conf_interval = stats.norm.interval(0.90, loc=mean_value, scale=np.std(data[column]))
    filtered_data = data[(data[column] >= conf_interval[0]) & (data[column] <= conf_interval[1])]
    
    if filtered_data.empty:
        filtered_data = data
    
    mean_filtered = filtered_data[column].mean()
    std_filtered = np.std(filtered_data[column]) / np.sqrt(len(filtered_data[column]))
    
    return mean_filtered, std_filtered



def plot_interactive_scatter(df, column, title=None, downsample=None, save_path=None):
  
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in DataFrame")

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    plot_df = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
    if downsample is not None and len(plot_df) > downsample:
        plot_df = plot_df.sample(downsample).sort_values("datetime")

    fig = px.scatter(
        plot_df,
        x="datetime",
        y=column,
        title=title or f"Scatter plot of {column}",
        labels={"datetime": "Datetime", column: column},
        opacity=0.7
    )

    # Format x-axis: avoid clustering
    fig.update_layout(
        xaxis=dict(
            tickformat="%Y-%m-%d",
            tickangle=30,
            showgrid=True,
        ),
        yaxis=dict(showgrid=True),
        margin=dict(l=60, r=20, t=60, b=60),
        template="plotly_white",
        hovermode="x unified"
    )

    if save_path:
        fig.write_html(save_path)

    fig.show()
    return fig


# --- main fitting fn ---
def fit_fb_data(data, specific_key, fb, train_end,val_end, window_size="30D", d=365.25, e=np.pi, min_points=0):

    all_band_values = {}
    total_samples = sum(len(grid_data) for grid_data in data[specific_key])
    
    if total_samples <= 100:
        return  # Exit if total samples are insufficient

    for grid_data in data[specific_key]:
        for band_key, band_data in grid_data.items():
            if band_key == fb: 
                if band_key in all_band_values:
                    all_band_values[band_key].append(band_data)
                else:
                    all_band_values[band_key] = [band_data]

    for band_key, band_values in all_band_values.items():
        # Extract datetime, Q3, and Max values and convert to DataFrame
        datetime_values = [pd.to_datetime(entry['datetime']) for entry in band_values]
        q3_values = [entry['Q3'] for entry in band_values]
        lat = [entry['lat'] for entry in band_values]
        lon = [entry['lon'] for entry in band_values]
        df_all= pd.DataFrame({
            'datetime': datetime_values,
            f'Q3_{band_key}': q3_values,
            'lat': lat,
            'lon':lon,
            # f'max_{band_key}':max_values

        })

        df_all.set_index('datetime', inplace=True)
        df_raw = df_all[df_all.index < val_end] # the data until the end of the window considerd for the training.
        col = f"Q3_{fb}"
        if col not in df_raw.columns:
            raise ValueError(f"{col} not in dataframe")

        # training slice
        df_train = df_raw[df_raw.index < train_end]
        # print('total training datal',len(df_train))
        # print('total total datal',len(df_raw))
        # Example: plot fb1 residuals
        # plot_interactive_scatter(df_raw, col, title="All data", downsample=5000)


        # resample
        resampled_groups = df_train.resample(window_size)
        results = []
        for _, dat in resampled_groups:
            dat = dat.dropna(subset=[col])
            if not dat.empty and len(dat) >= 2:
                mean, std = calculate_filtered_stats(dat, col)
                results.append({"window": dat.index[0],
                                f"mean_{col}": mean,
                                f"std_{col}": std})
        if len(results) < min_points:
            return None  # not enough data

        resampled_df = pd.DataFrame(results).dropna()
        # print('len of resampled data',len(resampled_df))
        # resampled_df2 = resampled_df.copy()
        # resampled_df2.set_index('window', inplace=True)
        # print(resampled_df)

        # numeric time for fit
                # Convert datetime index to numeric for fitting
        t_data = resampled_df['window']
        timestamp_data = pd.to_datetime(t_data)
        timestamp_values = timestamp_data.to_numpy().astype('datetime64[s]').astype('int')
        t = (timestamp_values - timestamp_values[0]) / (24 * 3600)  # convert to days
        q_vals = resampled_df[f"mean_{col}"].values
        q_std = np.where(resampled_df[f"std_{col}"].values == 0, 1e-8, resampled_df[f"std_{col}"].values)

        # initial guess
        init_guess = [np.mean(q_vals),
                    (q_vals[-1] - q_vals[0]) / len(q_vals),
                    (q_vals.max() - q_vals.min()) / 2,
                    d, e]

        # fit
        bounds = ([-np.inf, -np.inf, 0, 365.25, -np.inf],
                [np.inf, np.inf, 2.5, 365.2500001, np.inf])
        popt, _ = curve_fit(model_func, t, q_vals, p0=init_guess, sigma=q_std,
                            maxfev=650000, bounds=bounds)

        # apply fit to full df range

        timestamp_all = pd.to_datetime(df_raw.index)
        timestamp_values_all = timestamp_all.to_numpy().astype('datetime64[s]').astype('int')
        t_full = (timestamp_values_all - timestamp_values_all[0]) / (24 * 3600)  # convert to days

        fit_curve = model_func(t_full, *popt)
        fit_c_r = model_func(t, *popt)
        residuals = df_raw[col].values - fit_curve
        chi_square = np.sum(((q_vals- fit_c_r) / q_std) ** 2)
        dof = len(q_vals) - len(popt)
        rcs_resampled = chi_square / dof
        print(dof)
        print(rcs_resampled)
        # print("data points:", len(q_vals))
        # print("fit parameters:", len(popt))
        # # resampled_df2[f'fit_c_{col}'] = fit_c_r
        

        # update df
        df_out = df_raw.copy()
        df_out[f"fit_{col}"] = fit_curve
        df_out[f"Res_{col}"] = residuals
        # plot_interactive_scatter2(resampled_df2, col, title=f"fit-{specific_key}", downsample=None)
        # plot_interactive_scatter(df_out, f"Res_{col}", title=f"Residuals data-{specific_key}", downsample=None)

    return df_out,rcs_resampled



#  %%

df = []
filtered_keys = [key for key in data.keys() if 19 <= int(key[1:]) <= 145 ] #Why i named as limited grids
fbs = ['fb_1', 'fb_2', 'fb_3', 'fb_4', 'fb_5','fb_6', 'fb_7', 'fb_8', 'fb_9', 'fb_10','fb_11']  #'fb_2', 'fb_3', 'fb_4', 'fb_5','fb_7', 'fb_8', 'fb_9', 'fb_10', 

for i, w in enumerate(windows_train):
    print(f"Window {i}: Train {w['train_start']} → {w['train_end']}, "
          f"Val {w['val_start']} → {w['val_end']}")

    for key in filtered_keys:
        dfs = []
        for fb in fbs:
            print(key,fb)
            df1 = fit_fb_data(data, key, fb,w['train_end'],w['val_end'] , window_size="30D", d=365.25, e=np.pi, min_points=4)
            # print(df1.shape)
            if df1 is not None and not df1.empty:
                dfs.append(df1)
        if dfs:
            df_combined = pd.concat(dfs, axis=1)
            df.append(df_combined)

    if df:
        df_final = pd.concat(df)
        df_final.sort_index(inplace=True)
        # print(df_final)
        df_final=df_final.loc[:,~df_final.columns.duplicated()]
        df_final=df_final[df_final.index >= w['train_start']]
        # protocol=4 is safest across Python 3.4+ and old NumPy/Pandas
        df_final.to_pickle(
            f"{output_dir}/Background_data-window_{i}.pkl",
            protocol=4
        )

        print(df_final)
    else:
        print("No valid data to concatenate.")

# %%
