"""
scaling.py

This module provides utility functions to scale half-orbit time-series data
from a custom PyTorch Dataset (HalfOrbitPairDataset).

What it does:
-------------
- Collects all half-orbit sequences (x1 and x2) from a dataset.
- Fits a StandardScaler on the entire dataset (unsupervised normalization).
- Applies the fitted scaler to all sequences in training, validation, and test datasets.

Usage:
------
from lstm.scaling import scale_datasets

scaled_train, scaled_val, scaled_test = scale_datasets(train_dataset, val_dataset, test_dataset)
"""

# import torch
# from sklearn.preprocessing import StandardScaler

# def scale_datasets(train_dataset, val_dataset=None, test_dataset=None):
#     """
#     Scales the x1 and x2 pairs in the provided datasets using StandardScaler fitted on the training data.

#     Args:
#         train_dataset: Dataset containing training pairs (x1, x2).
#         val_dataset: Dataset containing validation pairs (optional).
#         test_dataset: Dataset containing test pairs (optional).

#     Returns:
#         Tuple of scaled datasets: (scaled_train, scaled_val, scaled_test)
#     """
#     # Collect all half-orbits from training data
#     all_half_orbits = []
#     for x1, x2 in train_dataset:
#         all_half_orbits.append(x1)
#         all_half_orbits.append(x2)

#     all_data = torch.cat(all_half_orbits).numpy()

#     # Fit scaler on training data
#     scaler = StandardScaler()
#     scaler.fit(all_data.reshape(-1, all_data.shape[-1]))
    
#     def apply_scaling(dataset):
#         if dataset is None:
#             return None
#         scaled_data = []
#         for x1, x2 in dataset:
#             x1_np = scaler.transform(x1.numpy())
#             x2_np = scaler.transform(x2.numpy())
#             scaled_data.append((torch.from_numpy(x1_np), torch.from_numpy(x2_np)))
#         return scaled_data

#     scaled_train = apply_scaling(train_dataset)
#     scaled_val = apply_scaling(val_dataset)
#     scaled_test = apply_scaling(test_dataset)

#     return scaled_train, scaled_val, scaled_test,scaler.mean_
import torch
from sklearn.preprocessing import StandardScaler

def scale_datasets(train_dataset_normal,train_dataset, val_dataset=None, test_dataset=None, fit =None):
    """
    Scales the x1 and x2 pairs in the provided datasets using StandardScaler fitted on the training data.
    Supports datasets that optionally return labels.
    
    Args:
        train_dataset: Dataset returning (x1, x2) or (x1, x2, label)
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
    
    Returns:
        Tuple: (scaled_train, scaled_val, scaled_test, scaler_mean)
    """
    # Collect all x1 and x2 from training set
    
    all_half_orbits = []
    for item in train_dataset_normal:
        x1, x2 = item[:2]  # support (x1, x2) or (x1, x2, label)
        all_half_orbits.extend([x1, x2])

    all_data = torch.cat(all_half_orbits).numpy()

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(all_data.reshape(-1, all_data.shape[-1]))
    

    def apply_scaling(dataset):
        if dataset is None:
            return None
        scaled_data = []
        for item in dataset:
            x1, x2 = item[:2]
            rest = item[2:]  # will be () if no label, (label,) if label present

            x1_scaled = scaler.transform(x1.numpy())
            x2_scaled = scaler.transform(x2.numpy())

            new_item = (torch.from_numpy(x1_scaled), torch.from_numpy(x2_scaled), *rest)
            scaled_data.append(new_item)
        return scaled_data

    scaled_train = apply_scaling(train_dataset)
    scaled_val = apply_scaling(val_dataset)
    scaled_test = apply_scaling(test_dataset)
    if fit:
        scaled_train = apply_scaling(train_dataset_normal)
    else:
        scaled_train = apply_scaling(train_dataset)

    return scaled_train, scaled_val, scaled_test, scaler.mean_


def scale_datasets_half_orbit(train_dataset_normal,train_dataset, val_dataset=None, test_dataset=None, fit =None):
    """
    Scales the x1 and x2 pairs in the provided datasets using StandardScaler fitted on the training data.
    Supports datasets that optionally return labels.
    
    Args:
        train_dataset: Dataset returning (x1, x2) or (x1, x2, label)
        val_dataset: Optional validation dataset
        test_dataset: Optional test dataset
    
    Returns:
        Tuple: (scaled_train, scaled_val, scaled_test, scaler_mean)
    """
    # Collect all x1 and x2 from training set
    
    scaler = StandardScaler()
    train_data = [data for data in train_dataset_normal]
    train_data = torch.cat(train_data).numpy()
    scaler.fit(train_data.reshape(-1, train_data.shape[-1]))

    

    def apply_scaling(dataset):
        if dataset is None:
            return None
        scaled_data = []
        for data in dataset:
            x = data.numpy()
            x = scaler.transform(x)
            new_item = x.reshape(data.shape)
            scaled_data.append(torch.from_numpy(new_item))
        return scaled_data

    scaled_train = apply_scaling(train_dataset)
    scaled_val = apply_scaling(val_dataset)
    scaled_test = apply_scaling(test_dataset)
    if fit:
        scaled_train = apply_scaling(train_dataset_normal)
    else:
        scaled_train = apply_scaling(train_dataset)

    return scaled_train, scaled_val, scaled_test, scaler.mean_