

import os
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_encoded_representation(
    model,
    data_loader,
    device=None,
    save_path=None,
    title="Latent Space",
    alpha=0.6,
    figsize=(8, 6),
    show_plot=True,
    return_latent=False,
    use_labels=False,
    colormap='tab10'  # for class coloring
):
    """
    Plots the 2D latent space from a model's encoder using the given data loader.

    Args:
        model (torch.nn.Module): Trained model with an encoder.
        data_loader (DataLoader): DataLoader returning (x1, x2) or (x1, x2, label).
        device (torch.device, optional): CUDA or CPU. Defaults to auto-detect.
        save_path (str, optional): Path to save the plot.
        title (str): Title of the plot.
        alpha (float): Transparency of scatter points.
        figsize (tuple): Figure size for the plot.
        show_plot (bool): Whether to show the plot after saving.
        return_latent (bool): Whether to return the latent representations.
        use_labels (bool): Whether the data_loader also returns labels.
        colormap (str): Matplotlib colormap for class visualization.

    Raises:
        ValueError: If latent space is less than 2D.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    latent_representations = []
    labels = []

    with torch.no_grad():
        for batch in data_loader:
            if use_labels:
                x1, x2, y = batch
                labels.extend(y.cpu().numpy())
                labels.extend(y.cpu().numpy())
            else:
                x1, x2= batch

            x1, x2 = x1.to(device), x2.to(device)
            z1 = model.encoder(x1)
            z2 = model.encoder(x2)
            latent_representations.append(z1)
            latent_representations.append(z2)

    latent_representations = torch.cat(latent_representations, dim=0)

    if latent_representations.size(1) < 2:
        raise ValueError("Latent representation must be at least 2-dimensional.")

    ld1 = latent_representations[:, 0].cpu().numpy()
    ld2 = latent_representations[:, 1].cpu().numpy()

    plt.style.use('default')
    plt.figure(figsize=figsize)

    if use_labels:
        labels = torch.tensor(labels)
        print(len(labels))
        colormap = ListedColormap(['blue', 'orange'])
        scatter = plt.scatter(ld1, ld2, c=labels, cmap=colormap, alpha=alpha)
        plt.legend(*scatter.legend_elements(), title="Class")
    else:
        plt.scatter(ld1, ld2, alpha=alpha, color='blue', label="Encoded Data")
        plt.legend()

    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved latent plot to: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    if return_latent:
        return latent_representations.cpu().numpy(), labels if use_labels else None
    return None
