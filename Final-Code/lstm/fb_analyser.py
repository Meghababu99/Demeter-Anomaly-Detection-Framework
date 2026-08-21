import os
import numpy as np
import matplotlib.pyplot as plt

class FBAnalyzer:
    def __init__(self, anomalies_set, seismic_sequences, sigma ,mean_percentage ,data_label,output_dir, model_id,p):
        """
        :param test_data: Dictionary containing 'anomalies_fb'
        :param seismic_sequences: List of lists, seismic sequences per FB_i
        :param output_dir: Directory to save the output plot
        :param model_id: Identifier for the model used (for filename)
        """
        self.anomalies = anomalies_set
        self.seismic_seqs_fb = seismic_sequences
        self.output_dir = output_dir
        self.model_id = model_id
        self.sigma = sigma
        self.mean_percentage = mean_percentage
        self.data_label=data_label
        self.p=p

    def fb_analysis(self):
        num_features = len(self.anomalies['anomalies_fb'])
        total_anomalies = [len(fb) for fb in self.anomalies['anomalies_fb']]
    

        model_errors = []
        model_values = []

        for i in range(num_features):
            total_sequences = len(self.anomalies['anomalies_fb'][i])
            anomalies = len(self.seismic_seqs_fb[i])  # seismic == anomalies
            p = anomalies / total_sequences if total_sequences > 0 else 0
            binomial_error = (np.sqrt(p * (1 - p) / total_sequences) * 100) if total_sequences > 0 else 0
            model_errors.append(binomial_error)
            model_value = (anomalies / total_sequences) * 100 if total_sequences > 0 else 0
            model_values.append(model_value)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(16, 8))
        x = np.arange(num_features)

        # Secondary y-axis: Total anomalies
        ax2 = ax1.twinx()
        ax2.bar(x, total_anomalies, alpha=0.3, color='red', label='Total Number of Anomalies', zorder=1)
        ax2.set_ylabel('Total Number of Anomalies', color='red', fontsize=15)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 260)

        # Primary y-axis: Seismic ratios with error bars
        ax1.errorbar(
            x, model_values, yerr=model_errors,
            fmt='o', ecolor='black', capsize=5, color='blue',
            label='Seismic Anomaly Ratio (with Binomial Error)', zorder=5
        )
        ax1.set_xlabel('Feature', fontsize=15)
        ax1.set_ylabel('Seismic Anomaly Ratio: P(S/A) (%)', color='blue', fontsize=15)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(38, 82)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'FB_{i+1}' for i in x])

        # Horizontal reference lines
        ax1.axhline(self.mean_percentage, color='red', linestyle='-', linewidth=1, label='Random Baseline Mean')
        ax1.axhline(self.mean_percentage + self.sigma, color='red', linestyle='--', linewidth=2, label='±1 σ Interval RST')
        ax1.axhline(self.mean_percentage - self.sigma, color='red', linestyle='--', linewidth=2)

        # Legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.title(f'Seismic Anomaly Ratio with Binomial Error and Total Anomalies Count: {self.data_label} Data', fontsize=15)
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        save_path = os.path.join(self.output_dir, f'FB_Siesmic_anomaly-{self.data_label}-model_{self.model_id}-p{self.p}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved to {save_path}")
