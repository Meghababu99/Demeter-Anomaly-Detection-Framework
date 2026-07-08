import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    RocCurveDisplay, ConfusionMatrixDisplay
)

    # plt.close()
from sklearn.metrics import roc_curve
# def evaluate_classifier(model, data_loader, output_dir, data_label, m, device=None, use_internal_classifier=True):
#     """
#     Evaluate a classifier (internal or external) using latent representations from an LSTM autoencoder.

#     Args:
#         model (nn.Module): Trained model (LSTM AE with or without internal classifier).
#         data_loader (DataLoader): DataLoader returning (x1, x2, y).
#         output_dir (str): Path to save evaluation metrics and plots.
#         data_label (str): Label for dataset (e.g., 'test', 'val').
#         m (int): Model version identifier for saving plots.
#         device (torch.device): Device to run inference on.
#         use_internal_classifier (bool): Whether to use model's internal classifier.
#     """
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     model.to(device)

#     os.makedirs(output_dir, exist_ok=True)

#     all_preds = []
#     all_probs = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in data_loader:
#             x1, x2, y = batch
#             x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()

#             if use_internal_classifier:
#                 _, out1 = model(x1)
#                 _, out2 = model(x2)
#             else:
#                 # External classifier scenario: encode, then classify
#                 z1 = model.encoder(x1)
#                 z2 = model.encoder(x2)
#                 raise NotImplementedError("External classifier logic not implemented.")

#             # Each x1, x2 pair has the same y
#             probs = torch.cat([out1, out2], dim=0).squeeze()
#             labels = torch.cat([y, y], dim=0)

#             preds = (probs >= 0.5).float()

#             all_probs.append(probs.cpu())
#             all_preds.append(preds.cpu())
#             all_labels.append(labels.cpu())

#     # Stack all batches
#     y_true = torch.cat(all_labels).numpy()
#     y_pred = torch.cat(all_preds).numpy()
#     y_prob = torch.cat(all_probs).numpy()

#     # Compute metrics
#     acc = accuracy_score(y_true, y_pred)
#     prec = precision_score(y_true, y_pred)
#     rec = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     auc = roc_auc_score(y_true, y_prob)
#     cm = confusion_matrix(y_true, y_pred)

#     print(f"\n[Evaluation Results] ({data_label})")
#     print(f"Accuracy     : {acc:.4f}")
#     print(f"Precision    : {prec:.4f}")
#     print(f"Recall       : {rec:.4f}")
#     print(f"F1 Score     : {f1:.4f}")
#     print(f"AUC-ROC      : {auc:.4f}")
#     print(f"Confusion Mat:\n{cm}")

#     # Save metrics
#     with open(os.path.join(output_dir, f"metrics_{data_label}_{m}.txt"), "w") as f:
#         f.write("Classifier Evaluation Metrics\n")
#         f.write(f"Accuracy     : {acc:.4f}\n")
#         f.write(f"Precision    : {prec:.4f}\n")
#         f.write(f"Recall       : {rec:.4f}\n")
#         f.write(f"F1 Score     : {f1:.4f}\n")
#         f.write(f"AUC-ROC      : {auc:.4f}\n")
#         f.write(f"Confusion Matrix:\n{cm}\n")

#     # Confusion matrix
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     fig, ax = plt.subplots()
#     disp.plot(ax=ax, cmap='Blues', colorbar=True)

#     # Access the colorbar and label it
#     cbar = ax.figure.axes[-1]  # Last axis is the colorbar
#     cbar.set_ylabel("Sample Count", rotation=270, labelpad=15)

#     plt.title(f"Confusion Matrix ({data_label} Data)")
#     plt.savefig(os.path.join(output_dir, f"confusion_matrix_{data_label}-{m}.png"))
#     plt.show()
#     plt.close()


#     # ROC Curve
#     RocCurveDisplay.from_predictions(y_true, y_prob)
#     plt.title(f"ROC Curve ({data_label})")
#     plt.savefig(os.path.join(output_dir, f"roc_curve_{data_label}-{m}.png"))
#     plt.show()
#     plt.close()

#     return {
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "auc": auc,
#         "confusion_matrix": cm
#     }

def evaluate_classifier(
    model_ae,
    latent_classifier,
    data_loader,
    output_dir,
    data_label,
    m,
    device=None,
    use_internal_classifier=True,
    threshold =0.5
    
):
    """
    Evaluate a classifier (internal or external) using latent representations from an LSTM autoencoder.

    Args:
        model (nn.Module): Trained model (e.g., LSTM AE with encoder and optional classifier).
        data_loader (DataLoader): DataLoader returning (x1, x2, y).
        output_dir (str): Path to save evaluation metrics and plots.
        data_label (str): Label for dataset (e.g., 'test', 'val').
        m (int): Model version identifier for saving plots.
        device (torch.device): Device to run inference on.
        use_internal_classifier (bool): Use model's internal classifier if True.
        latent_classifier (nn.Module): External classifier operating on latent vectors. Required if use_internal_classifier is False.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ae.eval()
    model_ae.to(device)

    if not use_internal_classifier:
        assert latent_classifier is not None, "Latent classifier must be provided for external classification."
        latent_classifier.eval()
        latent_classifier.to(device)

    os.makedirs(output_dir, exist_ok=True)

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()

            if use_internal_classifier:
                _, out1 = model_ae(x1)
                _, out2 = model_ae(x2)
                out1=torch.sigmoid(out1)
                out2=torch.sigmoid(out2)
            else:
                # External classifier: encode first, then classify
                z1 = model_ae.encoder(x1)
                z2 = model_ae.encoder(x2)
                out1 = latent_classifier(z1)
                print("before sig",out1)
                out1=torch.sigmoid(out1)
                print("after sig",out1)
                out2 = latent_classifier(z2)
                out2=torch.sigmoid(out2)

            probs = torch.cat([out1, out2], dim=0).squeeze()
            labels = torch.cat([y, y], dim=0)

            preds = (probs >= threshold).float()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Stack all batches
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)

    prec = precision_score(y_true, y_pred,zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[Evaluation Results] ({data_label})")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUC-ROC      : {auc:.4f}")
    print(f"Confusion Mat:\n{cm}")

    # Save metrics
    with open(os.path.join(output_dir, f"metrics_{data_label}-model_{m}-t{threshold*100}.txt"), "w") as f:
        f.write("Classifier Evaluation Metrics\n")
        f.write(f"Accuracy     : {acc:.4f}\n")
        f.write(f"Precision    : {prec:.4f}\n")
        f.write(f"Recall       : {rec:.4f}\n")
        f.write(f"F1 Score     : {f1:.4f}\n")
        f.write(f"AUC-ROC      : {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    cbar = ax.figure.axes[-1]
    cbar.set_ylabel("Sample Count", rotation=270, labelpad=15)

    plt.title(f"Confusion Matrix ({data_label} Data) -Threshold: {threshold}")
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{data_label}-model_{m}-t{threshold*100}.png"))
    plt.show()
    plt.close()

    # ROC Curve plot
    # RocCurveDisplay.from_predictions(y_true, y_prob)
    # plt.title(f"ROC Curve ({data_label} Data)")
    # plt.savefig(os.path.join(output_dir, f"roc_curve_{data_label}-model_{m}.png"))
    # plt.show()
    bin_size = 0.05
    data_min = min(y_pred.min(), y_prob.min())
    data_max = max(y_pred.max(), y_prob.max())
 
    bins = np.arange(data_min, data_max + bin_size, bin_size)


# Compute FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Scatter plot
    plt.figure()
    plt.scatter(fpr, tpr, color='blue', s=10, label=f'Scatter ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve as Scatter Plot ({data_label} Data)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"roc_scatter_{data_label}-model_{m}.png"))
    plt.show()
    plt.figure()
    plt.hist(y_pred[y_true==0], bins=bins, alpha=0.5, label='Non_seismic')
    plt.hist(y_pred[y_true==1], bins=bins, alpha=0.5, label='seismic')
    plt.legend()
    plt.title(f'Classifier - Prediction Distribution: Threshold : {threshold}')
    plt.xlabel('Discriminator output')
    plt.ylabel('Counts')
    plt.savefig(f'{output_dir}/Disciminator-{data_label}-modle_{m}-t{threshold*100}.png')
    plt.show()

    plt.figure()
    plt.hist(y_prob[y_true==0], bins=bins, alpha=0.5, label='Non_seismic')
    plt.hist(y_prob[y_true==1], bins=bins, alpha=0.5, label='seismic')
    plt.legend()
    plt.title(f'Classifier - Probabilities Distribution')
    plt.xlabel('Discriminator output')
    plt.ylabel('Counts')
    plt.savefig(f'{output_dir}/Disciminator_distribution-{data_label}-modle_{m}-t{threshold*100}.png')
    plt.show()

   

# how can i add this to my currnet fn

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm,
        "y_pred":y_pred,
        "y_true":y_true,
        "y_prob":y_prob
    }
