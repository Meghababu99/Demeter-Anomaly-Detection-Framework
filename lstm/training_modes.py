

import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_lstm_ae_mode(
    model,
    train_data_loader,
    val_data_loader,
    num_epochs=450,
    lr=0.001,
    patience=100,
    best_model_path="best_model.pth",
    early_stopping=True,
    save_all_epochs=False,
    all_epochs_dir=None,
    plot_loss=False,
    loss_plot_path=None,
    path2=None,
    device=None,
    input_size=None,
    hidden_size=None,
    latent_dim=None,
    num_layers=None,
    batch_size=None,
    seq=None,
    mode='autoencoder'  # 'autoencoder' or 'autoencoder_classifier'
):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_recon_losses, val_recon_losses = [], []
    train_class_losses, val_class_losses = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_recon_loss_epoch = 0.0
        train_class_loss_epoch = 0.0

        for batch in train_data_loader:
            optimizer.zero_grad()

            # Unpack batch
            if mode == 'autoencoder_classifier':
                x1, x2, y1 = batch
                y1 = y1.float().to(device)
            else:
                x1, x2 = batch

            x1, x2 = x1.to(device), x2.to(device)

            # Forward pass
            output1 = model(x1)
            output2 = model(x2)

            if mode == 'autoencoder_classifier':
                decoded1, class1 = output1
                decoded2, class2 = output2

                recon_loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
                class_loss = (bce_criterion(class1.squeeze(), y1) + bce_criterion(class2.squeeze(), y1)) / 2
                loss = 0.3 * recon_loss + 0.7 * class_loss

                train_recon_loss_epoch += recon_loss.item()
                train_class_loss_epoch += class_loss.item()
            else:
                decoded1 = output1
                decoded2 = output2
                loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
                train_recon_loss_epoch += loss.item()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_data_loader))
        train_recon_losses.append(train_recon_loss_epoch / len(train_data_loader))
        # scheduler.step()
        if mode == 'autoencoder_classifier':
            train_class_losses.append(train_class_loss_epoch / len(train_data_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        val_recon_loss_epoch = 0.0
        val_class_loss_epoch = 0.0

        with torch.no_grad():
            for batch in val_data_loader:
                if mode == 'autoencoder_classifier':
                    x1, x2, y1 = batch
                    y1 = y1.float().to(device)
                else:
                    x1, x2 = batch

                x1, x2 = x1.to(device), x2.to(device)

                output1 = model(x1)
                output2 = model(x2)

                if mode == 'autoencoder_classifier':
                    decoded1, class1 = output1
                    decoded2, class2 = output2

                    recon_loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
                    class_loss = (bce_criterion(class1.squeeze(), y1) + bce_criterion(class2.squeeze(), y1)) / 2
                    loss = 0.3 * recon_loss + 0.7 * class_loss

                    val_recon_loss_epoch += recon_loss.item()
                    val_class_loss_epoch += class_loss.item()
                else:
                    decoded1 = output1
                    decoded2 = output2
                    loss = (mse_criterion(decoded1, x1) + mse_criterion(decoded2, x2)) / 2
                    val_recon_loss_epoch += loss.item()

                val_loss += loss.item()

        val_losses.append(val_loss / len(val_data_loader))
        val_recon_losses.append(val_recon_loss_epoch / len(val_data_loader))
        if mode == 'autoencoder_classifier':
            val_class_losses.append(val_class_loss_epoch / len(val_data_loader))

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(all_epochs_dir, f"model_epoch_{epoch + 1}.pth"))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, best_model_path)

    # Plot overall loss
    if plot_loss and loss_plot_path:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs. Validation Loss')
        plt.legend()
        plt.grid(True)

        details = (
            f'Input Size: {input_size}\n'
            f'Hidden Size: {hidden_size}\n'
            f'Latent Dim: {latent_dim}\n'
            f'Num Layers: {num_layers}\n'
            f'Batch Size: {batch_size}\n'
            f'Learning Rate: {lr:.1e}\n'
            f'Sequence Length: 34\n'
            f'Mode: {mode}'
        )
        plt.text(0.5, 0.95, details, transform=plt.gca().transAxes,
         fontsize=9, va='top', ha='center',
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.6))

        os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
        plt.savefig(loss_plot_path, dpi=600, bbox_inches='tight')
        plt.show()

    # Plot separate recon/class loss if applicable
    if mode == 'autoencoder_classifier' and path2:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_recon_losses, label='Train Recon Loss')
        plt.plot(val_recon_losses, label='Val Recon Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.legend()
        plt.title('Reconstruction Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_class_losses, label='Train Class Loss')
        plt.plot(val_class_losses, label='Val Class Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.legend()
        plt.title('Classification Loss')

        plt.tight_layout()
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.show()

    return model, train_losses, val_losses



def train_lstm_ae_mode_half_orbit(
    model,
    train_data_loader,
    val_data_loader,
    num_epochs=450,
    lr=0.001,
    patience=100,
    best_model_path="best_model.pth",
    early_stopping=True,
    save_all_epochs=False,
    all_epochs_dir=None,
    plot_loss=False,
    loss_plot_path=None,
    path2=None,
    device=None,
    input_size=None,
    hidden_size=None,
    latent_dim=None,
    num_layers=None,
    batch_size=None,
    seq=None,
    mode='autoencoder'  # 'autoencoder' or 'autoencoder_classifier'
):
    """
    Training loop for single half-orbit case.

    Each batch contains:
        - For 'autoencoder': (x,)
        - For 'autoencoder_classifier': (x, y)

    mode controls training behavior:
        - 'autoencoder': train AE only, model returns decoded output.
        - 'autoencoder_classifier': train AE + classifier; model returns (decoded, class_pred).
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    train_losses, val_losses = [], []
    train_recon_losses, val_recon_losses = [], []
    train_class_losses, val_class_losses = [], []

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # ================================
        # TRAINING
        # ================================
        model.train()
        train_loss_epoch = 0.0
        train_recon_loss_epoch = 0.0
        train_class_loss_epoch = 0.0

        for batch in train_data_loader:
            optimizer.zero_grad()

            # Unpack batch
            if mode == "autoencoder_classifier":
                x, y = batch
                y = y.float().to(device)
            else:
                (x) = batch  # Dataset returns (x,)
            x = x.to(device)

            # Forward pass
            output = model(x)

            if mode == "autoencoder_classifier":
                decoded, class_pred = output
                recon_loss = mse_criterion(decoded, x)
                class_loss = bce_criterion(class_pred.squeeze(), y)
                loss = 0.3 * recon_loss + 0.7 * class_loss
                train_recon_loss_epoch += recon_loss.item()
                train_class_loss_epoch += class_loss.item()
            else:
                decoded = output
                loss = mse_criterion(decoded, x)
                train_recon_loss_epoch += loss.item()

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()

        # scheduler.step()
        train_losses.append(train_loss_epoch / len(train_data_loader))
        train_recon_losses.append(train_recon_loss_epoch / len(train_data_loader))
        if mode == "autoencoder_classifier":
            train_class_losses.append(train_class_loss_epoch / len(train_data_loader))

        # ================================
        # VALIDATION
        # ================================
        model.eval()
        val_loss_epoch = 0.0
        val_recon_loss_epoch = 0.0
        val_class_loss_epoch = 0.0

        with torch.no_grad():
            for batch in val_data_loader:
                if mode == "autoencoder_classifier":
                    x, y = batch
                    y = y.float().to(device)
                else:
                    (x) = batch
                x = x.to(device)

                output = model(x)

                if mode == "autoencoder_classifier":
                    decoded, class_pred = output
                    recon_loss = mse_criterion(decoded, x)
                    class_loss = bce_criterion(class_pred.squeeze(), y)
                    loss = 0.3 * recon_loss + 0.7 * class_loss
                    val_recon_loss_epoch += recon_loss.item()
                    val_class_loss_epoch += class_loss.item()
                else:
                    decoded = output
                    loss = mse_criterion(decoded, x)
                    val_recon_loss_epoch += loss.item()

                val_loss_epoch += loss.item()

        val_losses.append(val_loss_epoch / len(val_data_loader))
        val_recon_losses.append(val_recon_loss_epoch / len(val_data_loader))
        if mode == "autoencoder_classifier":
            val_class_losses.append(val_class_loss_epoch / len(val_data_loader))

        # ================================
        # LOGGING / EARLY STOP
        # ================================
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(all_epochs_dir, f"model_epoch_{epoch + 1}.pth"))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # ================================
    # SAVE BEST MODEL
    # ================================
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, best_model_path)

    # ================================
    # PLOTTING
    # ================================
    if plot_loss and loss_plot_path:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss (Single Half-Orbit Data Training)")
        plt.legend(loc='upper center')
        plt.grid(True)

        details = (
            f"Input Size: {input_size}\n"
            f"Hidden Size: {hidden_size}\n"
            f"Latent Dim: {latent_dim}\n"
            f"Num Layers: {num_layers}\n"
            f"Batch Size: {batch_size}\n"
            f"LR: {lr:.1e}\n"
            f"Seq Len: {seq}\n"
            f"Mode: {mode}"
        )
        plt.text(0.98, 0.95, details, transform=plt.gca().transAxes,
                 fontsize=9, va="top", ha="right",
                 bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3", alpha=0.6))
        os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
        plt.savefig(loss_plot_path, dpi=600, bbox_inches="tight")
        plt.show()

    # Separate recon/class loss curves if classifier mode
    if mode == "autoencoder_classifier" and path2:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_recon_losses, label="Train Recon Loss")
        plt.plot(val_recon_losses, label="Val Recon Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.title("Reconstruction Loss over Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_class_losses, label="Train Class Loss")
        plt.plot(val_class_losses, label="Val Class Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Classification Loss")
        plt.title("Classification Loss over Epochs")
        plt.legend()

        plt.tight_layout()
        os.makedirs(os.path.dirname(path2), exist_ok=True)
        plt.savefig(path2, dpi=600, bbox_inches="tight")
        plt.show()

    return model, train_losses, val_losses
