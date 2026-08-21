import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta
from shapely.geometry import Point
import torch


class SeismicAnalysis_wog:
    def __init__(
        self,
        dataset,
        earthquake_catalog,
        create_half_orbit_sequences,
        is_eq_fn,
        model_name = None,
        threshold_label=None,
        mean_rst = None,
        sigma_rst  = None,
        
        output_dir=None,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.dataset = dataset
        self.eq_catalog = earthquake_catalog
        self.create_half_orbit_sequences = create_half_orbit_sequences
        self.is_eq_fn = is_eq_fn

        self.m = model_name
        self.p = threshold_label
        self.mean_rst=mean_rst
        self.sigma_rst=sigma_rst

        # Internal cache for reuse
        self.datetime_sequences = None
        self.lat_long_sequences = None
        _, self.datetime_sequences, self.lat_long_sequences,_ = self.create_half_orbit_sequences(
            self.dataset)


    # Existing methods (agg_analysis, _plot_seismic_bar, etc.) remain unchanged below...
    def agg_analysis(self, data_label,anomalous_indices):
        plt.style.use('default')

        """Main analysis loop to classify anomalies and track matched earthquakes."""
        self.eq_catalog['Time'] = pd.to_datetime(self.eq_catalog['Time'])
        unique_earthquake_details = set()
        missed_eq = set()
        seismic_indices = []

        seismic_count = non_seismic_count = total = 0

        for idx in anomalous_indices:
            anomaly_time = self.datetime_sequences[idx][-1]
            sequence_labels = []

            for i, location in enumerate(self.lat_long_sequences[idx]):
                is_seismic, inside_spatial, outside_spatial = self.is_eq_fn(
                    anomaly_time,i, location, self.eq_catalog
                )

                if is_seismic:
                    for eq in inside_spatial:
                        unique_earthquake_details.add(tuple(eq.items()))
                else:
                    for eq in outside_spatial:
                        missed_eq.add(tuple(eq.items()))

                sequence_labels.append(is_seismic)
                total += 1

            if any(sequence_labels):
                seismic_count += 1
                seismic_indices.append(idx)
            else:
                non_seismic_count += 1
        missed_eq= missed_eq.difference(unique_earthquake_details)

        
            # Save CSV files for unique and missed earthquakes
        unique_eq_df = pd.DataFrame([dict(t) for t in unique_earthquake_details])
        
        missed_eq_df = pd.DataFrame([dict(t) for t in missed_eq])

        return seismic_indices, unique_eq_df, missed_eq
