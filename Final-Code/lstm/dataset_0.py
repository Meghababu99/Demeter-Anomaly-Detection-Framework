
"""
dataset.py

This module defines a PyTorch Dataset class for preparing paired half-orbit sequences 
from time-series data, typically satellite measurements.

What it does:
-------------
- Segments a DataFrame into non-overlapping half-orbits based on time gaps.
- Pairs adjacent half-orbits if the time gap is within a specified threshold.
- Optionally filters the pairs based on labels (e.g., use only pairs with label == 0).
- Converts the paired sequences into PyTorch tensors, ready for use in DataLoaders.

Usage:
------
from lstm import HalfOrbitPairDataset

dataset = HalfOrbitPairDataset(df, df_labels, use_label_0_only=True)
"""


import torch
from torch.utils.data import Dataset
import pandas as pd

class HalfOrbitPairDataset(Dataset):

    def __init__(self, df, df_labels, min_data_points=17, transform=None, use_label_0_only=False, return_labels=False):
        self.transform = transform
        self.min_data_points = min_data_points
        self.df = df
        self.df_labels = df_labels
        self.features = df.filter(regex='^Res_Q3_').columns
        self.use_label_0_only = use_label_0_only
        self.return_labels = return_labels

        self.label_map = dict(zip(df_labels['index'], df_labels['label']))
        self.label_0_pair_indices = set(df_labels[df_labels['label'] == 0]['index'])

        self.all_pairs = self.extract_half_orbit_pairs(df)
        self.pairs = self.filter_pairs() if use_label_0_only else self.all_pairs

    def extract_half_orbit_pairs(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")

        df = df.sort_index()

        gap_threshold = pd.Timedelta(hours=1)
        max_pair_gap = pd.Timedelta(hours=2)

        gaps = df.index.to_series().diff() > gap_threshold
        gap_indices = gaps[gaps].index
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]

        half_orbits = []

        for i in range(0, len(segment_boundaries) - 1, 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            segment = df.loc[(df.index >= start) & (df.index < end), self.features]

            if len(segment) >= self.min_data_points:
                half_orbits.append((segment[:self.min_data_points].values, start))

        pairs = []
        for i in range(0, len(half_orbits) - 1, 2):
            seq1, time1 = half_orbits[i]
            seq2, time2 = half_orbits[i + 1]

            time_diff = time2 - time1
            if time_diff <= max_pair_gap:
                pairs.append((seq1, seq2))

        return pairs

    def filter_pairs(self):
        return [pair for idx, pair in enumerate(self.all_pairs) if idx in self.label_0_pair_indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        if self.return_labels:
            label = self.label_map.get(idx, -1)  # default to -1 if label not found
            return x1, x2, torch.tensor(label, dtype=torch.long)

        return x1, x2
    
    def create_half_orbit_sequences(self,df,non_eq_pair_indices=None):
        """
        Generate half-orbit paired sequences and optionally filter pairs by provided indices.

        Args:
            df (pd.DataFrame): Input DataFrame with DateTimeIndex and features matching 'Res_Q3_'.
            min_data_points (int): Minimum number of data points required for a valid sequence.
            label_0_pair_indices (set or list, optional): Indices of pairs to keep. If None, keep all pairs.

        Returns:
            tuple: 
                - sequences: List of combined half-orbit sequences.
                - datetime_sequences: List of corresponding datetime sequences.
                - lat_long_sequences: List of corresponding latitude-longitude sequences.
                - sequence_info: List of tuples (pair_index, len_seq1, len_seq2, len_combined)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")
        
        df = df.sort_index()
        
        sequences, datetime_sequences, lat_long_sequences = [], [], []
        seq1_lengths, seq2_lengths, combined_lengths = [], [], []
        sequence_info = []

        features = df.filter(regex='^Res_Q3_').columns
        lat_col, long_col = 'lat', 'lon'
        
        gap_threshold = pd.Timedelta(hours=1)
        max_pair_gap = pd.Timedelta(hours=2)
        
        gaps = df.index.to_series().diff() > gap_threshold
        gap_indices = gaps[gaps].index
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]
        
        half_orbits = []
        time = []
        lat_long = []

        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            segment = df.loc[(df.index >= start) & (df.index < end), features]
            dt = df.loc[(df.index >= start) & (df.index < end)].index
            latlong = df.loc[(df.index >= start) & (df.index < end), [lat_col, long_col]]
            if len(segment) >= self.min_data_points:
                half_orbits.append((segment[:self.min_data_points], start, end))
                time.append(dt[:self.min_data_points])
                lat_long.append(latlong[:self.min_data_points])
        
        pair_idx = 0
        for i in range(0, len(half_orbits) - 1, 2):
            seq1, t1_start, t1_end = half_orbits[i]
            seq2, t2_start, t2_end = half_orbits[i + 1]
            dt1 = time[i]
            dt2 = time[i + 1]
            lat_long1 = lat_long[i]
            lat_long2 = lat_long[i + 1]

            time_gap = t2_start - t1_start
            if time_gap <= max_pair_gap:
                # Only append if no filtering or pair_idx in label_0_pair_indices
                if (non_eq_pair_indices is None) or (pair_idx in non_eq_pair_indices):
                    combined_seq = pd.concat([seq1, seq2])
                    combined_dt = dt1.append(dt2)
                    combined_lat_long = pd.concat([lat_long1, lat_long2])
                    
                    sequences.append(combined_seq.values)
                    datetime_sequences.append(combined_dt.values)
                    lat_long_sequences.append(combined_lat_long.values)
                    
                    seq1_lengths.append(len(seq1))
                    seq2_lengths.append(len(seq2))
                    combined_lengths.append(len(combined_seq))
                    
                    sequence_info.append((pair_idx, len(seq1), len(seq2), len(combined_seq)))
                pair_idx += 1

        print(f"[INFO] Total valid combined sequences: {len(sequences)}")
        
        length_df = pd.DataFrame({
            'Seq1 Length': seq1_lengths,
            'Seq2 Length': seq2_lengths,
            'Combined Length': combined_lengths
        })
        
        return sequences, datetime_sequences, lat_long_sequences, sequence_info
