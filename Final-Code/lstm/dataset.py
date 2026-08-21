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

# Without labels:
dataset = HalfOrbitPairDataset(df)

# With labels (expects columns: 'index' (pair index) and 'label'):
dataset = HalfOrbitPairDataset(df, df_labels, use_label_0_only=True, return_labels=True)
"""

from typing import Optional, Iterable, Tuple, List
import torch
from torch.utils.data import Dataset
import pandas as pd


class HalfOrbitPairDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        df_labels: Optional[pd.DataFrame] = None,
        min_data_points: int = 17,
        transform=None,
        use_label_0_only: bool = False,
        return_labels: bool = False,
        feature_regex: str = r"^Res_Q3_",
        lat_col: str = "lat",
        lon_col: str = "lon",
        gap_threshold_hours: float = 1.0,
        max_pair_gap_hours: float = 2.0,
    ):
        """
        Args:
            df: DataFrame indexed by DateTimeIndex with feature columns (default matching '^Res_Q3_').
            df_labels: Optional DataFrame with columns ['index', 'label'] mapping pair index -> label.
            min_data_points: Minimum points per half-orbit segment.
            transform: Optional transform to apply to each tensor sequence.
            use_label_0_only: If True, keep only pairs whose label == 0 (requires df_labels).
            return_labels: If True, __getitem__ returns (x1, x2, label). If labels are missing, label=-1.
            feature_regex: Regex to select feature columns.
            lat_col / lon_col: Column names for lat/lon (used in create_half_orbit_sequences).
            gap_threshold_hours: Time gap that splits half-orbits.
            max_pair_gap_hours: Max allowed time gap between paired half-orbits.
        """
        self.transform = transform
        self.min_data_points = int(min_data_points)
        self.df = df
        self.df_labels = df_labels
        self.use_label_0_only = bool(use_label_0_only)
        self.return_labels = bool(return_labels)
        self.lat_col = lat_col
        self.lon_col = lon_col

        # Time thresholds
        self.gap_threshold = pd.Timedelta(hours=float(gap_threshold_hours))
        self.max_pair_gap = pd.Timedelta(hours=float(max_pair_gap_hours))

        # Select features
        self.features = df.filter(regex=feature_regex).columns
        if len(self.features) == 0:
            raise ValueError(
                f"No feature columns match regex '{feature_regex}'. "
                "Pass a different 'feature_regex' or ensure your DataFrame has the expected columns."
            )

        # Label-related structures (safe if df_labels is None)
        if df_labels is not None:
            required_cols = {"index", "label"}
            missing = required_cols.difference(df_labels.columns)
            if missing:
                raise ValueError(f"df_labels is missing required columns: {missing}")
            self.label_map = dict(zip(df_labels["index"], df_labels["label"]))
            self.label_0_pair_indices = set(df_labels.loc[df_labels["label"] == 0, "index"])
        else:
            self.label_map = {}  # empty map
            self.label_0_pair_indices = set()

        # Build all pairs from df
        self.all_pairs: List[Tuple[Iterable, Iterable]] = self.extract_half_orbit_pairs(self.df)

        # Apply optional filtering
        if self.use_label_0_only:
            if df_labels is None:
                print("[WARN] use_label_0_only=True but df_labels=None; keeping all pairs.")
                self.pairs = self.all_pairs
            else:
                self.pairs = self.filter_pairs()
        else:
            self.pairs = self.all_pairs

    def extract_half_orbit_pairs(self, df: pd.DataFrame):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")

        df = df.sort_index()

        gaps = df.index.to_series().diff() > self.gap_threshold
        gap_indices = gaps[gaps].index
        # Boundaries: [first, ...gaps..., last]
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]

        half_orbits: List[Tuple[pd.DataFrame, pd.Timestamp]] = []

        for i in range(0, len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            # left-closed, right-open to avoid double-counting boundaries
            mask = (df.index >= start) & (df.index < end)
            segment = df.loc[mask, self.features]
            if len(segment) >= self.min_data_points:
                half_orbits.append((segment.iloc[: self.min_data_points].values, start))

        # Pair adjacent half-orbits (0-1, 2-3, ...)
        pairs = []
        for i in range(0, len(half_orbits) - 1, 2):
            seq1, time1 = half_orbits[i]
            seq2, time2 = half_orbits[i + 1]
            time_diff = time2 - time1
            if time_diff <= self.max_pair_gap:
                pairs.append((seq1, seq2))

        return pairs

    def filter_pairs(self):
        # Keep only pairs whose index is in label_0_pair_indices
        return [pair for idx, pair in enumerate(self.all_pairs) if idx in self.label_0_pair_indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        x1, x2 = self.pairs[idx]
        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)

        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        if self.return_labels:
            # If labels not provided, return -1
            label = self.label_map.get(idx, -1)
            return x1, x2, torch.tensor(label, dtype=torch.long)

        return x1, x2

    def create_half_orbit_sequences(
        self,
        df: pd.DataFrame,
        non_eq_pair_indices: Optional[Iterable[int]] = None,
    ):
        """
        Generate half-orbit paired sequences and optionally filter pairs by provided indices.

        Args:
            df (pd.DataFrame): Input DataFrame with DateTimeIndex and features matching feature_regex.
            non_eq_pair_indices (Iterable[int] | None): Indices of pairs to keep. If None, keep all pairs.

        Returns:
            tuple:
                - sequences: List[np.ndarray] of concatenated feature sequences (pair-wise).
                - datetime_sequences: List[np.ndarray] of corresponding timestamps.
                - lat_long_sequences: List[np.ndarray] of corresponding [lat, lon].
                - sequence_info: List[tuple] -> (pair_index, len_seq1, len_seq2, len_combined)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")

        df = df.sort_index()

        features = df.loc[:, self.features]
        if self.lat_col not in df.columns or self.lon_col not in df.columns:
            raise ValueError(
                f"Expected latitude/longitude columns '{self.lat_col}' and '{self.lon_col}' in df."
            )

        sequences, datetime_sequences, lat_long_sequences = [], [], []
        seq1_lengths, seq2_lengths, combined_lengths = [], [], []
        sequence_info = []

        gaps = df.index.to_series().diff() > self.gap_threshold
        gap_indices = gaps[gaps].index
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]

        half_orbits = []
        times = []
        lat_lons = []

        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            mask = (df.index >= start) & (df.index < end)

            seg_feats = df.loc[mask, self.features]
            seg_time = df.loc[mask].index
            seg_latlon = df.loc[mask, [self.lat_col, self.lon_col]]

            if len(seg_feats) >= self.min_data_points:
                half_orbits.append((seg_feats.iloc[: self.min_data_points], start, end))
                times.append(seg_time[: self.min_data_points])
                lat_lons.append(seg_latlon.iloc[: self.min_data_points])

        pair_idx = 0
        for i in range(0, len(half_orbits) - 1, 2):
            seq1, t1_start, _ = half_orbits[i]
            seq2, t2_start, _ = half_orbits[i + 1]
            dt1 = times[i]
            dt2 = times[i + 1]
            latlon1 = lat_lons[i]
            latlon2 = lat_lons[i + 1]

            time_gap = t2_start - t1_start
            if time_gap <= self.max_pair_gap:
                if (non_eq_pair_indices is None) or (pair_idx in set(non_eq_pair_indices)):
                    combined_seq = pd.concat([seq1, seq2])
                    combined_dt = dt1.append(dt2)
                    combined_latlon = pd.concat([latlon1, latlon2])

                    sequences.append(combined_seq.values)
                    datetime_sequences.append(combined_dt.values)
                    lat_long_sequences.append(combined_latlon.values)

                    seq1_lengths.append(len(seq1))
                    seq2_lengths.append(len(seq2))
                    combined_lengths.append(len(combined_seq))

                    sequence_info.append((pair_idx, len(seq1), len(seq2), len(combined_seq)))
                pair_idx += 1

        print(f"[INFO] Total valid combined sequences: {len(sequences)}")

        length_df = pd.DataFrame(
            {
                "Seq1 Length": seq1_lengths,
                "Seq2 Length": seq2_lengths,
                "Combined Length": combined_lengths,
            }
        )
        # Optionally return length_df if you want to inspect lengths externally
        return sequences, datetime_sequences, lat_long_sequences, sequence_info


class HalfOrbitDataset(Dataset):
    """
    Dataset for single half-orbit sequences (no pairing).
    Segments a time-indexed DataFrame into half-orbits based on time gaps.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_labels: Optional[pd.DataFrame] = None,
        min_data_points: int = 17,
        transform=None,
        use_label_0_only: bool = False,
        return_labels: bool = False,
        feature_regex: str = r"^Res_Q3_",
        lat_col: str = "lat",
        lon_col: str = "lon",
        gap_threshold_hours: float = 1.0,

    ):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")

        self.df = df.sort_index()
        self.transform = transform
        self.min_data_points = int(min_data_points)
        self.gap_threshold = pd.Timedelta(hours=float(gap_threshold_hours))
        self.df_labels = df_labels
        self.use_label_0_only = bool(use_label_0_only)
        self.return_labels = bool(return_labels)
        self.lat_col, self.lon_col= lat_col,lon_col 
        # Select features
        self.features = df.filter(regex=feature_regex).columns
        if len(self.features) == 0:
            raise ValueError(
                f"No feature columns match regex '{feature_regex}'. "
                "Pass a different 'feature_regex' or ensure your DataFrame has the expected columns."
            )

           # Label-related structures (safe if df_labels is None)
        if df_labels is not None:
            required_cols = {"index", "label"}
            missing = required_cols.difference(df_labels.columns)
            if missing:
                raise ValueError(f"df_labels is missing required columns: {missing}")
            self.label_map = dict(zip(df_labels["index"], df_labels["label"]))
            self.label_0_pair_indices = set(df_labels.loc[df_labels["label"] == 0, "index"])
        else:
            self.label_map = {}  # empty map
            self.label_0_pair_indices = set()

        # Build all pairs from df
        self.all_sequences: List[Iterable]= self.extract_half_orbits(self.df)

        # Apply optional filtering
        if self.use_label_0_only:
            if df_labels is None:
                print("[WARN] use_label_0_only=True but df_labels=None; keeping all pairs.")
                self.sequences = self.all_sequences
            else:
                self.sequences = self.filter_pairs()
        else:
            self.sequences= self.all_sequences



    def extract_half_orbits(self, df: pd.DataFrame):
        df = self.df
        gaps = df.index.to_series().diff() > self.gap_threshold
        gap_indices = gaps[gaps].index

        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]
        # print(segment_boundaries)
        half_orbits = []
        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            mask = (df.index >= start) & (df.index < end)
            segment = df.loc[mask, self.features]
            if len(segment) >= self.min_data_points:
                half_orbits.append(segment.iloc[: self.min_data_points].values)

        return half_orbits
    def filter_pairs(self):
    # Keep only pairs whose index is in label_0_pair_indices
        return [sequences for idx, sequences in enumerate(self.all_sequences) if idx in self.label_0_pair_indices]
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x
    
    
    def create_half_orbit_sequences(self, df: pd.DataFrame):
        """
        Generate single half-orbit sequences and accompanying metadata (timestamps, lat/lon).

        Args:
            df (pd.DataFrame): Input DataFrame with DateTimeIndex and features matching feature_regex.

        Returns:
            tuple:
                - sequences: List[np.ndarray] of feature sequences.
                - datetime_sequences: List[np.ndarray] of corresponding timestamps.
                - lat_long_sequences: List[np.ndarray] of corresponding [lat, lon].
                - sequence_info: List[tuple] -> (segment_index, length)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")

        df = df.sort_index()

        # Validate lat/lon columns
        if self.lat_col not in df.columns or self.lon_col not in df.columns:
            raise ValueError(
                f"Expected latitude/longitude columns '{self.lat_col}' and '{self.lon_col}' in df."
            )

        features = df.loc[:, self.features]

        sequences, datetime_sequences, lat_long_sequences = [], [], []
        lengths, sequence_info = [], []

        # Detect time gaps that define orbit boundaries
        gaps = df.index.to_series().diff() > self.gap_threshold
        gap_indices = gaps[gaps].index
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]

        segment_idx = 0
        for i in range(len(segment_boundaries) - 1):
            start, end = segment_boundaries[i], segment_boundaries[i + 1]
            mask = (df.index >= start) & (df.index < end)

            seg_feats = df.loc[mask, self.features]
            seg_time = df.loc[mask].index
            seg_latlon = df.loc[mask, [self.lat_col, self.lon_col]]

            if len(seg_feats) >= self.min_data_points:
                seg_feats = seg_feats.iloc[: self.min_data_points]
                seg_time = seg_time[: self.min_data_points]
                seg_latlon = seg_latlon.iloc[: self.min_data_points]

                sequences.append(seg_feats.values)
                datetime_sequences.append(seg_time.values)
                lat_long_sequences.append(seg_latlon.values)
                lengths.append(len(seg_feats))
                sequence_info.append((segment_idx, len(seg_feats)))
                segment_idx += 1

        print(f"[INFO] Total valid half-orbit sequences: {len(sequences)}")

        length_df = pd.DataFrame({"Length": lengths})
        # Optionally return length_df externally if needed

        return sequences, datetime_sequences, lat_long_sequences, sequence_info

