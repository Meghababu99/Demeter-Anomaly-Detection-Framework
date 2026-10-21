# anomaly.py

import torch
import numpy as np
import matplotlib.pyplot as plt


class AnomalyDetector:
    def __init__(self, model, dataloader=None, criterion=None, num_features=None, anomaly_thresholds=None, device=None):
        self.model = model.eval()
        self.dataloader = dataloader
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.num_features = num_features
        self.anomaly_thresholds = anomaly_thresholds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if anomaly_thresholds is not None and num_features is not None:
            assert len(anomaly_thresholds) == num_features, "Thresholds must match number of features"

    def to_device(self, x):
        return x.to(self.device)

    def compute_reconstruction_errors_agg(self, dataloader):
        errors = []
        with torch.no_grad():
            for x1, x2 in dataloader:
                x1, x2 = self.to_device(x1), self.to_device(x2)
                out1, out2 = self.model(x1), self.model(x2)
                loss1 = self.criterion(out1, x1)
                loss2 = self.criterion(out2, x2)
                errors.append(((loss1 + loss2) / 2).item())
        return np.array(errors)

    def compute_reconstruction_errors_fb(self, dataloader):
        errors = []
        with torch.no_grad():
            for x1, x2 in dataloader:
                x1, x2 = self.to_device(x1), self.to_device(x2)
                out1, out2 = self.model(x1), self.model(x2)
                err1 = ((out1 - x1) ** 2).mean(dim=1)
                err2 = ((out2 - x2) ** 2).mean(dim=1)
                batch_error = ((err1 + err2) / 2).cpu().numpy()
                errors.extend(batch_error)
        return np.array(errors)

    def detect_anomalies_agg(self, errors, threshold):
        return [i for i, e in enumerate(errors) if e >= threshold]

    def detect_anomalies_fb(self, errors, thresholds):
        num_features = errors.shape[1]
        anomalous_indices = [[] for _ in range(num_features)]
        for i, row in enumerate(errors):
            for j in range(num_features):
                if row[j] >= thresholds[j]:
                    anomalous_indices[j].append(i)
        return anomalous_indices

    def plot_reconstruction_error(self, errors, title, threshold=None,percentile=None, save_path=None):
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title(title)
        if threshold:
            plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {percentile}%')
            plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_feature_errors(self, errors_fb, title=None, threshold=None,percentile=None,  save_path=None):

        num_features = 11
        plt.figure(figsize=(12, 6))
        for i in range(num_features):
            plt.subplot(3, 4, i + 1)
            plt.hist(errors_fb[:, i], bins=50, edgecolor='black')
            plt.xlabel('Error', fontsize=10)
            plt.ylabel('Frequency', fontsize=10)
            plt.title(f'Feature {i+1}', fontsize=10)
            if threshold is not None:
                plt.axvline(threshold[i], color='r', linestyle='--', label=f'Threshold: {percentile}%')
                plt.legend(fontsize=7)
            plt.xticks(fontsize=9)
            plt.yticks(fontsize=9)
            plt.grid(True)
        plt.suptitle(title or 'Feature-wise Reconstruction Error')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()




class AnomalyDetector_half_orbit:
    def __init__(self, model, dataloader=None, criterion=None, num_features=None, anomaly_thresholds=None, device=None):
        self.model = model.eval()
        self.dataloader = dataloader
        self.criterion = criterion if criterion is not None else torch.nn.MSELoss()
        self.num_features = num_features
        self.anomaly_thresholds = anomaly_thresholds
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if anomaly_thresholds is not None and num_features is not None:
            assert len(anomaly_thresholds) == num_features, "Thresholds must match number of features"

    def to_device(self, x):
        return x.to(self.device)

    def compute_reconstruction_errors_agg(self, dataloader):
        errors = []
        with torch.no_grad():
            for x1 in dataloader:
                x1 = self.to_device(x1)
                out1 = self.model(x1)
                loss1 = self.criterion(out1, x1)
                errors.append(loss1.item())
        return np.array(errors)

    def compute_reconstruction_errors_fb(self, dataloader):
        errors = []
        with torch.no_grad():
            for x1 in dataloader:
                x1 = self.to_device(x1)
                out1= self.model(x1)
                err1 = ((out1 - x1) ** 2).mean(dim=1)
                batch_error = err1.cpu().numpy()
                errors.extend(batch_error)
        return np.array(errors)

    def detect_anomalies_agg(self, errors, threshold):
        return [i for i, e in enumerate(errors) if e >= threshold]

    def detect_anomalies_fb(self, errors, thresholds):
        num_features = errors.shape[1]
        anomalous_indices = [[] for _ in range(num_features)]
        for i, row in enumerate(errors):
            for j in range(num_features):
                if row[j] >= thresholds[j]:
                    anomalous_indices[j].append(i)
        return anomalous_indices

    # def plot_reconstruction_error(self, errors, title, threshold=None,percentile=None, save_path=None):
    #     plt.figure(figsize=(10, 6))
    #     plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    #     plt.xlabel('Reconstruction Error')
    #     plt.ylabel('Frequency')
    #     plt.title(title)
    #     if threshold:
    #         plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {percentile}%')
    #         plt.legend()
    #     plt.grid(True)
    #     if save_path:
    #         plt.savefig(save_path, dpi=300)
    #     plt.show()

    # def plot_feature_errors(self, errors_fb, title=None, threshold=None,percentile=None,  save_path=None):

    #     num_features = 11
    #     plt.figure(figsize=(12, 6))
    #     for i in range(num_features):
    #         plt.subplot(3, 4, i + 1)
    #         plt.hist(errors_fb[:, i], bins=50, edgecolor='black')
    #         plt.xlabel('Error', fontsize=10)
    #         plt.ylabel('Frequency', fontsize=10)
    #         plt.title(f'Feature {i+1}', fontsize=10)
    #         if threshold is not None:
    #             plt.axvline(threshold[i], color='r', linestyle='--', label=f'Threshold: {percentile}%')
    #             plt.legend(fontsize=7)
    #         plt.xticks(fontsize=9)
    #         plt.yticks(fontsize=9)
    #         plt.grid(True)
    #     plt.suptitle(title or 'Feature-wise Reconstruction Error')
    #     plt.tight_layout()
        
    #     if save_path:
    #         plt.savefig(save_path, dpi=300)
    #     plt.show()

