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


class SeismicAnalysis:
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
    def analyze_anomalies_per_feature(
        self,
        data_label,anomalous_indices,
        plot: bool = True,
    ):
        """
        Analyze reconstruction errors to find anomalies per feature,
        correlate with earthquake catalog, and optionally plot or save results.
        """
        self.eq_catalog['Time'] = pd.to_datetime(self.eq_catalog['Time'])
    
        num_features = 11  # Assuming fixed feature count; could parameterize if needed

        # Create half orbit sequences (timestamps and locations)
        

        anomalous_timestamps_per_feature = [[] for _ in range(num_features)]
        anomalous_loc_per_feature = [[] for _ in range(num_features)]

        # Collect timestamps and locations for anomalies per feature
        for fidx in range(num_features):
            for idx in anomalous_indices[fidx]:
                anomalous_timestamps_per_feature[fidx].append(self.datetime_sequences[idx])
                anomalous_loc_per_feature[fidx].append(self.lat_long_sequences[idx])

        anomaly_labels_per_feature = [[] for _ in range(num_features)]
        unique_earthquake_details_features = set()
        missed_eq_fb = set()

        # Label anomalies as seismic or not and collect matching EQ info
        for fidx in range(num_features):
            for seq_idx in range(len(anomalous_timestamps_per_feature[fidx])):
                seq_times = anomalous_timestamps_per_feature[fidx][seq_idx]
                seq_locs = anomalous_loc_per_feature[fidx][seq_idx]
                seq_labels = []

                timestamp = seq_times[-1]

                for pt_idx in range(len(seq_times)):
                    location = seq_locs[pt_idx]
                    is_seismic, inside_spatial, outside_spatial = self.is_eq_fn(
                        timestamp, location, self.eq_catalog
                    )
                    if is_seismic:
                        for eq in inside_spatial:
                            unique_earthquake_details_features.add(tuple(eq.items()))
                    else:
                        for eq in outside_spatial:
                            missed_eq_fb.add(tuple(eq.items()))

                    seq_labels.append(is_seismic)

                anomaly_labels_per_feature[fidx].append(seq_labels)

        # Remove earthquakes from missed that are already in unique
        missed_eq_fb = missed_eq_fb.difference(unique_earthquake_details_features)

        
            # Save CSV files for unique and missed earthquakes
        unique_eq_df = pd.DataFrame([dict(t) for t in unique_earthquake_details_features])
        missed_eq_df = pd.DataFrame([dict(t) for t in missed_eq_fb])

        # unique_eq_df.to_csv(
        #     os.path.join(self.output_dir, f"FB-UniqueEQ-{data_label}-model_{self.m}-p{self.p}.csv"),
        #     index=False,
        # )
        # missed_eq_df.to_csv(
        #     os.path.join(self.output_dir, f"FB-MissedEQ-{data_label}-model_{self.m}-p{self.p}.csv"),
        #     index=False,
        # )
        sequence_stats_per_feature = {feature_idx: {'total': 0, 'anomalies': 0} for feature_idx in range(num_features)}

        for feature_idx in range(num_features):
            for seq_idx, idx in enumerate(anomalous_indices[feature_idx]):
    
                anomaly_labels = anomaly_labels_per_feature[feature_idx][seq_idx]
                sequence_stats_per_feature[feature_idx]['total'] += 1  # total anomalous sequence count
                if sum(anomaly_labels) >= 1:
                    sequence_stats_per_feature[feature_idx]['anomalies'] += 1 

        seismic_seqs_per_feature = {f: set() for f in range(num_features)}
        for feature_idx in range(num_features):
            for seq_idx, idx in enumerate(anomalous_indices[feature_idx]):
                anomaly_labels = anomaly_labels_per_feature[feature_idx][seq_idx]
                if sum(anomaly_labels) >= 1:
                    seismic_seqs_per_feature[feature_idx].add(idx)
        if plot:
            # Plot feature-wise reconstruction error histograms
            sequence_stats_per_feature = {f: {'total': 0, 'anomalies': 0} for f in range(num_features)}
    
            for feature_idx in range(num_features):
                for seq_idx, idx in enumerate(anomalous_indices[feature_idx]):
                    anomaly_labels = anomaly_labels_per_feature[feature_idx][seq_idx]
                    sequence_stats_per_feature[feature_idx]['total'] += 1
                    if sum(anomaly_labels) >= 1:
                        sequence_stats_per_feature[feature_idx]['anomalies'] += 1
            
            # Plot anomaly ratios per feature
            plt.figure(figsize=(20, 14))
            legend_handles = []
            
            for i in range(num_features):
                total_sequences = sequence_stats_per_feature[i]['total']
                anomalies = sequence_stats_per_feature[i]['anomalies']
                anomaly_percentage = (anomalies / total_sequences) * 100 if total_sequences > 0 else 0
                
                ax = plt.subplot(3, 4, i + 1)
                bar_total = ax.bar(0, total_sequences, color='skyblue', label='Total Sequences')
                bar_anomalies = ax.bar(0, anomalies, color='orange', label='Seismic Sequences')
                ax.text(-0.3, anomalies / 2, f'{anomaly_percentage:.2f}%', fontsize=15, ha='center', va='center')
                ax.text(-0.3, 3 * total_sequences / 4, '100%', fontsize=15, ha='center', va='bottom')
                ax.text(0.3, 3 * total_sequences / 4, f'#{total_sequences}', fontsize=15, ha='center', va='bottom')
                ax.text(0.3, anomalies / 2, f'#{anomalies}', fontsize=15, ha='center', va='bottom')
                ax.text(0, anomalies / 2, 'Seismic', fontsize=15, ha='center', va='center')
                
                mean_y = self.mean_rst / 100 * total_sequences
                upper_y = (self.mean_rst + self.sigma_rst) / 100 * total_sequences
                lower_y = (self.mean_rst - self.sigma_rst) / 100 * total_sequences
                
                ax.axhline(y=mean_y, color='red', linestyle='-', linewidth=2)
                ax.axhline(y=upper_y, color='red', linestyle='--', linewidth=2)
                ax.axhline(y=lower_y, color='red', linestyle='--', linewidth=2)
                
                if i == 0:
                    legend_handles.extend([bar_total[0], bar_anomalies[0], ax.lines[0], ax.lines[1]])
                
                plt.xticks([0], [''])
                ax.set_xlabel('Sequence Label')
                ax.set_ylabel('Count')
                ax.grid(True)
                ax.set_title(f'Feature FB {i + 1}')
            
            # plt.legend(handles=legend_handles, labels=['Total Sequences', 'Seismic Sequences', 'Mean-RST', '�1 s RST'], 
            #         loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)
            plt.suptitle(f'Seismic Anomaly Ratio - FB Analysis (98% Threshold): {data_label} data', fontsize=15)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/FB-seismic_Anomaly_ratio-{data_label}-model_{self.m}-p{self.p}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Correlation matrices of anomalies
            anomalous_indices_sets = [set(indices) for indices in anomalous_indices]
            anomalous_correlation_matrix = np.zeros((num_features, num_features))
            for i in range(num_features):
                for j in range(num_features):
                    anomalous_correlation_matrix[i, j] = len(anomalous_indices_sets[i].intersection(anomalous_indices_sets[j]))
            
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(anomalous_correlation_matrix, annot=True, cmap='coolwarm', fmt='g',
                        xticklabels=[f'Fb{i+1}' for i in range(num_features)],
                        yticklabels=[f'Fb{i+1}' for i in range(num_features)], annot_kws={"size": 12})
            plt.title(f'Anomalous Indices Correlation Matrix - FB Analysis: {data_label} data')
            plt.xlabel('Feature Band')
            plt.ylabel('Feature Band')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/FB-Anomalous-CorMatrix-{data_label}-model_{self.m}-p{self.p}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        
            
            common_indices_matrix = np.zeros((num_features, num_features), dtype=int)
            for i in range(num_features):
                for j in range(i, num_features):
                    common = seismic_seqs_per_feature[i].intersection(seismic_seqs_per_feature[j])
                    common_indices_matrix[i, j] = len(common)
                    common_indices_matrix[j, i] = len(common)
            
            plt.figure(figsize=(8, 6))
            sns.set(font_scale=1.2)
            sns.heatmap(common_indices_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=True,
                        xticklabels=[f'fb{i+1}' for i in range(num_features)],
                        yticklabels=[f'fb{i+1}' for i in range(num_features)],
                        annot_kws={"size": 10})
            plt.title(f'Seismic Anomaly Indices Correlation Matrix - FB Analysis: {data_label} Data')
            plt.xlabel('Feature Band')
            plt.ylabel('Feature Band')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/FB-Common-seismic_seq-{data_label}-model_{self.m}-p{self.p}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close
            print(seismic_seqs_per_feature)
            seismic_indices = [
                                item for sublist in seismic_seqs_per_feature.values()
                                if isinstance(sublist, (list, set)) for item in sublist
                            ]
            self._plot_event_counts(data_label,unique_earthquake_details_features,tag='FB')
            self.plot_seismic_time_distribution(data_label, seismic_indices,tag='FB')
            # self.plot_matched_earthquake_locations( data_label,unique_earthquake_details_features,tag='Agg')
    
            # Other plots could be added here if needed, or you can call class plotting methods.

        # Return the same outputs as your original function
        return (
     
            seismic_seqs_per_feature,
            unique_earthquake_details_features,
            missed_eq_fb
        )

    # Existing methods (agg_analysis, _plot_seismic_bar, etc.) remain unchanged below...
    def agg_analysis(self, data_label,anomalous_indices,plot=True):
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

            for location in self.lat_long_sequences[idx]:
                is_seismic, inside_spatial, outside_spatial = self.is_eq_fn(
                    anomaly_time, location, self.eq_catalog
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

        # unique_eq_df.to_csv(
        #     os.path.join(self.output_dir, f"Agg_UniqueEQ-{data_label}-model_{self.m}-p{self.p}.csv"),
        #     index=False,
        # )
        # missed_eq_df.to_csv(
        #     os.path.join(self.output_dir, f"Agg_MissedEQ-{data_label}-model_{self.m}-p{self.p}.csv"),
        #     index=False,
        # )
        if plot:

            self._plot_seismic_bar(data_label,seismic_count, non_seismic_count)
            self._plot_event_counts(data_label,unique_earthquake_details,tag='Agg')
            # self.plot_matched_earthquake_locations( data_label,unique_earthquake_details,tag='Agg')
            self.plot_seismic_time_distribution(data_label, seismic_indices,tag='Agg')

        return seismic_indices, unique_eq_df, missed_eq

    def _plot_seismic_bar(self,data_label,seismic_count, non_seismic_count):
        plt.style.use('default')
        total = seismic_count + non_seismic_count
        s_pct = (seismic_count / total) * 100
        ns_pct = (non_seismic_count / total) * 100

        plt.figure(figsize=(8, 6))
        plt.bar(['Total'], [seismic_count], color='orange', label='Seismic')
        plt.bar(['Total'], [non_seismic_count], bottom=[seismic_count], color='skyblue', label='Non-Seismic')

        plt.text(0, seismic_count / 2, f'Seismic: {seismic_count}\n({s_pct:.1f}%)', ha='center')
        plt.text(0, seismic_count + non_seismic_count / 2, f'Non-Seismic: {non_seismic_count}\n({ns_pct:.1f}%)', ha='center')
        plt.title(f'Seismic Anomaly Ratio - Agg Analysis (98% Threshold) : {data_label} Data')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Agg-seismic_Anomaly_ratio-{data_label}-model_{self.m}-p{self.p}.png", dpi=300)
        plt.close()

    def _plot_event_counts(self,data_label, correlated_eqs,tag):
        plt.style.use('default')
        test_start = self.dataset.index[0]
        test_end = self.dataset.index[-1] + timedelta(days=2)

        eq_in_test_range = self.eq_catalog[(self.eq_catalog['Time'] >= test_start) & (self.eq_catalog['Time'] <= test_end)]
        total_count = len(eq_in_test_range)
        found = len(correlated_eqs)
        not_found = total_count - found

        found_pct = (found / total_count) * 100 if total_count > 0 else 0
        not_found_pct = 100 - found_pct

        plt.figure(figsize=(8, 6))
        plt.bar(['Matched', 'UnMatched'], [found, not_found], color=['lightcoral', 'skyblue'])

        plt.text(0, found, f'{found} ({found_pct:.1f}%)', ha='center')
        plt.text(1, not_found / 2, f'{not_found} ({not_found_pct:.1f}%)', ha='center')
        plt.title(f'Matched Earthquakes - {tag} Analysis : {data_label} Data')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{tag}-EQ_Matching-{data_label}-model_{self.m}-p{self.p}.png", dpi=300)
        plt.show()
        plt.close()

    def plot_seismic_time_distribution(self,data_label, seismic_indices,tag):
        plt.style.use('default')
        """Plot monthly histogram of seismic anomaly timestamps."""
        if not seismic_indices or self.datetime_sequences is None:
            print("Missing data to plot time distribution.")
            return

        seismic_times = [self.datetime_sequences[idx][-1] for idx in seismic_indices]
        df = pd.DataFrame({'Timestamp': pd.to_datetime(seismic_times)})
        df['Month'] = df['Timestamp'].dt.to_period('M').dt.to_timestamp()

        plt.figure(figsize=(10, 6))
        sns.histplot(df['Month'], bins=len(df['Month'].unique()), kde=False, color='tomato')
        plt.title(f'Seismic Anomaly Time Distribution -{tag} Analysis: {data_label} Data')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{tag}-Seismic_Timeline_{data_label}-model_{self.m}-p{self.p}.png", dpi=300)
        plt.close()

    def plot_matched_earthquake_locations(self, data_label,correlated_eqs,tag):
        plt.style.use('default')
        """Plot geographic locations of matched earthquakes."""
        if not correlated_eqs:
            print("No correlated earthquakes to plot.")
            return

        eq_list = [dict(eq) for eq in correlated_eqs]
        eq_df = pd.DataFrame(eq_list)

        geometry = [Point(xy) for xy in zip(eq_df['long'], eq_df['lat'])]
        gdf = gpd.GeoDataFrame(eq_df, geometry=geometry)

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        plt.figure(figsize=(12, 8))
        world.plot(ax=plt.gca(), color='lightgrey')
        gdf.plot(ax=plt.gca(), color='crimson', markersize=30, label='Matched Earthquake')

        plt.title(f'Matched Earthquake Locations - {tag} Analysis')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{tag}-Matched_EQ_Locations_{data_label}-model_{self.m}-p{self.p}.png", dpi=300)
        plt.close()
