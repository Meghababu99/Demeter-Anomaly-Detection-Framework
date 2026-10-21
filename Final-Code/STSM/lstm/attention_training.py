"""Paired temporal-attention AE training, following training_modes.py.

Retains Adam, paired mean MSE, validation, early stopping, per-epoch weights,
loss plotting and the (model, train_losses, val_losses) return convention.
No classifier/BCE/KL/attention supervision is added: attention learns through
backpropagation of the same reconstruction loss as the original AE.
Existing training modules are not imported or changed.
"""

from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def train_attention_lstm_ae_mode(
    model,
    train_data_loader,
    val_data_loader,
    num_epochs=450,
    lr=0.001,
    patience=100,
    best_model_path='best_temporal_attention_model.pth',
    early_stopping=True,
    save_all_epochs=False,
    all_epochs_dir=None,
    plot_loss=False,
    loss_plot_path=None,
    device=None,
    input_size=None,
    hidden_size=None,
    latent_dim=None,
    num_layers=None,
    batch_size=None,
    seq=None,
    mode='temporal_attention',
):
    """Each loader batch must be (X_A, X_B), trained on non-seismic pairs.

    L_A = mean((reconstruction_A - X_A)**2), over batch, time and features.
    L_B is defined identically; optimization loss = (L_A + L_B) / 2.
    Both forward calls use the SAME model and therefore share all weights.
    Epoch curves retain the original arithmetic mean of batch losses.

    Reliability differences confined to this new file: clone the best state
    (state_dict alone is a live view), save it immediately for Colab persistence,
    guard empty loaders/nonfinite losses, and allow a bare checkpoint filename.
    These .pth files are model weights, not optimizer/RNG states: loading them
    warm-starts training; it does not resume an interrupted epoch exactly.
    """
    if mode != 'temporal_attention' or getattr(model, 'model_type', None) != mode:
        raise ValueError('Use TemporalAttentionLSTMAutoencoder and mode="temporal_attention"')
    if num_epochs < 1 or len(train_data_loader) == 0 or len(val_data_loader) == 0:
        raise ValueError('Positive num_epochs and nonempty train/validation loaders required')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dtype = next(model.parameters()).dtype
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    os.makedirs(os.path.dirname(os.fspath(best_model_path)) or '.', exist_ok=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x1, x2 in train_data_loader:
            optimizer.zero_grad()
            x1, x2 = x1.to(device=device, dtype=dtype), x2.to(device=device, dtype=dtype)
            decoded1 = model(x1)
            decoded2 = model(x2)
            # Same objective as original paired AE; gradients also train attention.
            loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss: check input/scaling')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_data_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2 in val_data_loader:
                x1, x2 = x1.to(device=device, dtype=dtype), x2.to(device=device, dtype=dtype)
                decoded1 = model(x1)
                decoded2 = model(x2)
                loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite validation loss: check input/scaling')
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_data_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(all_epochs_dir, f'model_epoch_{epoch + 1}.pth'))
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if plot_loss and loss_plot_path:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs. Validation Loss (temporal_attention)')
        plt.legend()
        plt.grid(True)
        details = (
            f'Input Size: {input_size}\nHidden Size: {hidden_size}\n'
            f'Latent Dim: {latent_dim}\nNum Layers: {num_layers}\n'
            f'Batch Size: {batch_size}\nLearning Rate: {lr:.1e}\n'
            f'Sequence Length: {seq}\nMode: {mode}'
        )
        plt.text(0.5, 0.95, details, transform=plt.gca().transAxes,
                 fontsize=9, va='top', ha='center',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.6))
        os.makedirs(os.path.dirname(os.fspath(loss_plot_path)) or '.', exist_ok=True)
        plt.savefig(loss_plot_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
    return model, train_losses, val_losses
