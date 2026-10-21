import pandas as pd
import numpy as np
import os
from lstm import SeismicCriteria

class SeismicRatioEvaluator:
    def __init__(self, dataset, Anomalies_set,create_half_orbit_sequences,data_label, eq_data, time_window_values, output_dir, model_id,p):
        self.create_half_orbit_sequences=create_half_orbit_sequences
        self.sequences, self.timestamps, self.locations, _ = self.create_half_orbit_sequences(dataset)
        # print("total data",len(self.sequences))
        self.anomalies_agg = Anomalies_set['anomalies_agg']
        # print("total agg anomalies",len(Anomalies_set['anomalies_agg']))
        self.anomalies_fb = Anomalies_set['anomalies_fb']
        self.eq_data = eq_data
        self.time_window_values = time_window_values
        self.output_dir = output_dir
        self.model_id = model_id
        # self.ratios_by_window = {tw: [] for tw in time_window_values}
        self.data_label = data_label
        self.p=p

    def compute_agg_ratios(self):
        results = []

        for time_window in self.time_window_values:
            seismic_criteria = SeismicCriteria(spatial_width=20, time_window_hours=time_window)

            total_anomalous_sequences = len(self.anomalies_agg)
            total_seismic_sequences = 0

            selected_sequences = [self.sequences[idx] for idx in self.anomalies_agg]
            selected_timestamps = [self.timestamps[idx] for idx in self.anomalies_agg]
            selected_locations = [self.locations[idx] for idx in self.anomalies_agg]

            for seq_idx in range(total_anomalous_sequences):
                timestamps = selected_timestamps[seq_idx]
                locations = selected_locations[seq_idx]
                sequence_labels = []

                timestamp = pd.to_datetime(timestamps[-1])

                for point_idx in range(len(timestamps)):
                    location = locations[point_idx]
                    label, _, _ = seismic_criteria.is_eq(timestamp, location, self.eq_data)
                    sequence_labels.append(label)

                if any(sequence_labels):
                    total_seismic_sequences += 1

            ratio = total_seismic_sequences / total_anomalous_sequences if total_anomalous_sequences > 0 else 0
            # print("ratio", ratio)
            # print('time_window', time_window)
            results.append({"Time Window": time_window, "Seismic Ratio": ratio})

        # Create DataFrame once after loop
        df = pd.DataFrame(results)

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(
            self.output_dir,
            f"seismic_ratios-aggregate_20-{self.data_label}-model_{self.model_id}-p{self.p}.csv"
        )
        df.to_csv(file_path, index=False)
        print(f"Final aggregate data saved to {file_path}")


    def compute_fb_ratios(self):
        ratios_by_window = {tw: [] for tw in self.time_window_values}
        for i in range(len(self.anomalies_fb)):
            anomaly_indices = self.anomalies_fb[i]
            selected_sequences = [self.sequences[idx] for idx in anomaly_indices]
            selected_timestamps = [self.timestamps[idx] for idx in anomaly_indices]
            selected_locations = [self.locations[idx] for idx in anomaly_indices]

            seismic_ratios_for_csv = []

            for time_window in self.time_window_values:
                seismic_criteria = SeismicCriteria(spatial_width=20, time_window_hours=time_window)

                total_anomalous_sequences = len(selected_sequences)
                total_seismic_sequences = 0

                for seq_idx in range(total_anomalous_sequences):
                    timestamps = selected_timestamps[seq_idx]
                    locations = selected_locations[seq_idx]
                    sequence_labels = []

                    timestamp = pd.to_datetime(timestamps[-1])

                    for point_idx in range(len(timestamps)):
                        location = locations[point_idx]
                        label, _, _ = seismic_criteria.is_eq(timestamp, location, self.eq_data)
                        sequence_labels.append(label)

                    if any(sequence_labels):
                        total_seismic_sequences += 1

                seismic_ratio = total_seismic_sequences / total_anomalous_sequences if total_anomalous_sequences > 0 else 0
                ratios_by_window[time_window].append(seismic_ratio)

                seismic_ratios_for_csv.append({
                    'Time Window': time_window,
                    'Seismic Ratio': seismic_ratio
                })

            df = pd.DataFrame(seismic_ratios_for_csv)
            os.makedirs(self.output_dir, exist_ok=True)
            csv_path = os.path.join(self.output_dir, f"seismic_ratios-20-{self.data_label}-model_{self.model_id}-p{self.p}-fb_{i}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")

