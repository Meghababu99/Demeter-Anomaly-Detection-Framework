import os
import numpy as np
import plotly.graph_objects as go
from PIL import Image
  

class SeismicAnomalyAnalyzer:
    def __init__(self, dataset, eq,is_eq_func, percentiles,threshold_value,num_features,filename_suffix, create_half_orbit_sequences_func, output_dir="output",m='Model_number'):
        """
        Initializes the SeismicAnomalyAnalyzer.

        Args:
            is_eq_func (callable): Function that checks if a time/location matches a seismic event.
            create_half_orbit_sequences_func (callable): Function that returns sequences from the dataset.
            output_dir (str): Path where plots and summaries will be saved.
             percentiles (List[int]): Percentile bins (e.g., [0, 20, 40, 60, 80, 100]).
            dataset (Any): Dataset from which sequences will be generated.
            eq (Any): Earthquake catalog.
            filename_suffix (str): Suffix for output file naming.
            time_window (float): Time window (in days) to consider as matching a seismic event.
            spatial_window (float): Spatial window (in km) for seismic proximity.
            threshold_value (float): Value (in %) to mark as a threshold on the bar plot.
        """
        self.dataset=dataset
        self.percentiles=percentiles
        # self.time_window=time_window
        # self.spatial_window=spatial_window
        self.threshold_value=threshold_value
        self.filename_suffix=filename_suffix
        self.eq=eq
        self.is_eq = is_eq_func
        self.num_features = num_features
        self.create_half_orbit_sequences = create_half_orbit_sequences_func
        self.output_dir = output_dir
        self.m=m
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_aggregate_error_analysis(self, error_values):
        """
        Analyzes a single vector of anomaly scores (e.g., total reconstruction errors) 
        and evaluates the seismic ratio per percentile bin.

        Args:
            error_values (np.ndarray): 1D array of anomaly scores.
           
        """
        test_sequences, test_datetime_sequences, lat_long_sequences, _ = self.create_half_orbit_sequences(self.dataset)

        anoms_train_lim = [(p, np.percentile(error_values, p)) for p in self.percentiles]
        bin_results = []

        for i, (pct, lim) in enumerate(anoms_train_lim[:-1]):
            next_lim = anoms_train_lim[i + 1][1]
            bin_indices = [idx for idx, error in enumerate(error_values) if lim <= error < next_lim]
            total = len(bin_indices)
            seismic_count = 0

            for idx in bin_indices:
                sequence_labels = []
                anomaly_time = test_datetime_sequences[idx][-1]
                for location in lat_long_sequences[idx]:
                    is_seismic, *_ = self.is_eq(anomaly_time, location, self.eq)
                    sequence_labels.append(is_seismic)
                if any(sequence_labels):
                    seismic_count += 1

            seismic_ratio = (seismic_count / total) * 100 if total > 0 else 0
            bin_results.append({
                "bin": f"{pct}-{self.percentiles[i + 1]}%",
                "total_anomalies": total,
                "seismic_anomalies": seismic_count,
                "seismic_ratio": seismic_ratio
            })
        file_base = f"Aggregate_seismic_anomaly_{self.filename_suffix}-model_{self.m}"
        html_path = os.path.join(self.output_dir, f"{file_base}.html")
        png_path = os.path.join(self.output_dir, f"{file_base}.png")
        self._generate_bar_plot(
            bin_results,
            f"Seismic Anomaly Ratio per Percentile Bin : {self.filename_suffix.capitalize()} Data",
            html_file=None,
            png_file=png_path
        )

    def plot_feature_error_analysis(self, error_matrix):
        """
        Evaluates the seismic anomaly ratio for each feature independently.

        Args:
            error_matrix (np.ndarray): 2D array of reconstruction errors (rows = sequences, cols = features).
            percentiles (List[int]): Percentile thresholds for binning.
            dataset (Any): Dataset to analyze.
            eq (Any): Earthquake catalog.
            num_features (int): Number of features (columns) in the matrix.
            filename_suffix (str): Suffix for saving output plots.
            time_window (float): Time window (in days) for checking proximity to earthquakes.
            spatial_window (float): Spatial range (in km) to consider.
            threshold_value (float): Reference line in plots as a % of max value.
        """
        test_sequences, test_datetime_sequences, lat_long_sequences, _ = self.create_half_orbit_sequences(self.dataset)

        for feature_idx in range(self.num_features):
            errors = error_matrix[:, feature_idx]
            anoms_train_lim = [(p, np.percentile(errors, p)) for p in self.percentiles]
            bin_results = []

            for i, (pct, lim) in enumerate(anoms_train_lim[:-1]):
                next_lim = anoms_train_lim[i + 1][1]
                bin_indices = [idx for idx, error in enumerate(errors) if lim <= error < next_lim]
                total = len(bin_indices)
                seismic_count = 0

                for idx in bin_indices:
                    sequence_labels = []
                    anomaly_time = test_datetime_sequences[idx][-1]
                    for location in lat_long_sequences[idx]:
                        is_seismic, *_ = self.is_eq(anomaly_time, location, self.eq)
                        sequence_labels.append(is_seismic)
                    if any(sequence_labels):
                        seismic_count += 1

                seismic_ratio = (seismic_count / total) * 100 if total > 0 else 0
                bin_results.append({
                    "bin": f"{pct}-{self.percentiles[i + 1]}%",
                    "total_anomalies": total,
                    "seismic_anomalies": seismic_count,
                    "seismic_ratio": seismic_ratio
                })

            file_base = f"seismic_anomaly_feature_{feature_idx}_{self.filename_suffix}-model_{self.m}"
            html_path = os.path.join(self.output_dir, f"{file_base}.html")
            png_path = os.path.join(self.output_dir, f"{file_base}.png")
            self._generate_bar_plot(
                bin_results,
                f"Seismic Anomaly Ratio per Percentile Bin: {self.filename_suffix.capitalize()} - Feature {feature_idx+1}",
                html_file=None,
                png_file=png_path
            )

    def _generate_bar_plot(self, bin_results, title, html_file=None, png_file=None):
        """
        Creates a bar plot visualizing total and seismic anomalies per percentile bin.

        Args:
            bin_results (List[dict]): Data containing bin labels and counts. if for reconstruction_fb- the input will be specific for each fb
            title (str): Plot title.
            html_file (str): Path to save the HTML plot.
            png_file (str, optional): Optional path to save PNG image.
            threshold_value (float, optional): Draws a horizontal threshold line at % of max count.
        """
        bins = [r["bin"] for r in bin_results]
        totals = [r["total_anomalies"] for r in bin_results]
        seismics = [r["seismic_anomalies"] for r in bin_results]
        ratios = [r["seismic_ratio"] for r in bin_results]


        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins, y=totals, name='Total Anomalies', marker_color='lightgrey'))
        fig.add_trace(go.Bar(x=bins, y=seismics, name='Seismic Anomalies', marker_color='crimson'))

        max_y = max(totals)
        threshold_y = self.threshold_value * max_y / 100
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(bins) - 0.5,
            y0=threshold_y,
            y1=threshold_y,
            line=dict(color='RoyalBlue', width=2, dash='dash'),
            name=f'Mean_RST: {self.threshold_value }%',
            showlegend=True
        )

        for i, ratio in enumerate(ratios):
            fig.add_annotation(
                x=bins[i],
                y=max_y + 5,
                text=f"{ratio:.1f}%",
                showarrow=False,
                font=dict(color="black", size=8)
            )

        fig.update_layout(
            title=title,
            xaxis_title="Reconstruction Error Percentile Bin",
            yaxis_title="Number of Anomalies",
            barmode='overlay',
            bargap=0.2
        )
        fig.write_image(png_file,width=1200, height=800, scale=2)

        if html_file:

            fig.write_html(html_file)
        
 

   