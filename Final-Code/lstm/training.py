

"""
Training loop for LSTM Autoencoder models with optional early stopping and every epoch saving.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Training loop for LSTM Autoencoder models with optional early stopping,
model saving per epoch, and training loss plot.
"""

def train_lstm_ae(
    model, train_data_loader, val_data_loader,
    num_epochs=450, lr=0.001, patience=100, best_model_path="best_model.pth",
    early_stopping=True, save_all_epochs=False, all_epochs_dir=None,
    plot_loss=False, loss_plot_path=None,
    device=None,
    input_size=None, hidden_size=None, latent_dim=None, num_layers=None, batch_size=None, seq=None
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for x1, x2 in train_data_loader:
            x1, x2 = x1.to(device), x2.to(device)

            out1 = model(x1)
            out2 = model(x2)

            loss1 = criterion(out1, x1)
            loss2 = criterion(out2, x2)
            loss = (loss1 + loss2) / 2

            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss / len(train_data_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x1, x2 in val_data_loader:
                x1, x2 = x1.to(device), x2.to(device)

                out1 = model(x1)
                out2 = model(x2)

                loss1 = criterion(out1, x1)
                loss2 = criterion(out2, x2)
                loss = (loss1 + loss2) / 2
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_data_loader))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        # Save model for every epoch if requested
        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)
            epoch_model_path = os.path.join(all_epochs_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), epoch_model_path)

        # Early stopping logic
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # Load and save best model if early stopping was used or best state tracked
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, best_model_path)

    # Plot and save loss curve if requested
    if plot_loss and loss_plot_path:
        plt.style.use('default')
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        details = (
            f'Input Size: {input_size}\n'
            f'Hidden Size: {hidden_size}\n'
            f'Latent Dim: {latent_dim}\n'
            f'Num Layers: {num_layers}\n'
            f'Batch Size: {batch_size}\n'
            f'Learning Rate: {lr:.1e}\n'
            f'Sequence Length: {seq+seq} data points'
        )

        plt.text(0.98, 0.95, details, transform=plt.gca().transAxes,
                 fontsize=9, va='top', ha='right',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.6))

        plt.title('Train and Validation Loss Curves')
        plt.legend(fontsize=8)
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    return model, train_losses, val_losses

