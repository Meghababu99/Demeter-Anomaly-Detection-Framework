import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lstm import LSTMAutoencoder, LatentClassifier

class LatentClassifierTrainer:
    def __init__(
        self,         # Class reference to LSTMAutoencoder
        model_ae_pth,
        input_size=None,        # Used for plotting
        hidden_size=None,
        latent_dim=None,
        num_layers=None,
        batch_size=None,
        seq=None,
        combine_method=None,  # Options: "mean" or "concat"
        lr=None,
        patience=None,
        early_stopping=True,
        best_model_path="best_latent_model.pth",
        save_all_epochs=False,
        all_epochs_dir=None,
        plot_loss=True,
        loss_plot_path=None):
        """
        Trainer for classifier using latent representations from a pre-trained LSTM autoencoder.

        Args:
            model_pth (str): Path to the trained LSTMAutoencoder.
            classifier (nn.Module): Classifier model.
            combine_method (str): 'mean' or 'concat'.
            lr (float): Learning rate.
            patience (int): Patience for early stopping.
            early_stopping (bool): Whether to use early stopping.
            best_model_path (str): Path to save the best model.
            save_all_epochs (bool): Save model at each epoch.
            all_epochs_dir (str): Directory to save all epoch checkpoints (used only if save_all_epochs=True).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and freeze AE encoder
        autoencoder = LSTMAutoencoder(input_size, hidden_size, num_layers, latent_dim,use_internal_classifier=False)
        autoencoder.load_state_dict(torch.load(model_ae_pth, map_location=self.device))
        autoencoder.eval()
        self.encoder = autoencoder.encoder.to(self.device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.seq = seq
        # Classifier setup
    
        classifier = LatentClassifier(latent_dim = self.latent_dim)
        self.classifier = classifier.to(self.device)
        # Settings
        self.combine_method = combine_method
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr)
        self.patience = patience
        self.early_stopping = early_stopping
        self.best_model_path = best_model_path
        self.save_all_epochs = save_all_epochs
        self.all_epochs_dir = all_epochs_dir
        self.plot_loss = plot_loss
        self.loss_plot_path = loss_plot_path

       

        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)

    def _process_batch(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1,z2
        # if self.combine_method == "mean":
        #     return (z1 + z2) / 2
        # elif self.combine_method == "concat":
        #     return torch.cat([z1, z2], dim=1)
        # else:
        #     raise ValueError("combine_method must be 'mean' or 'concat'.")

    def train(self, train_loader, val_loader=None, num_epochs=50):
        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            self.classifier.train()
            total_loss = 0.0

            for x1, x2, y in train_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device).float()

                with torch.no_grad():
                    z1,z2 = self._process_batch(x1, x2)

                output1 = self.classifier(z1).squeeze()
                output2 = self.classifier(z2).squeeze()
                loss = (self.criterion(output1, y) + self.criterion(output2, y)) / 2

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if val_loader:
                self.classifier.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x1, x2, y in val_loader:
                        x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device).float()
                        z1,z2 = self._process_batch(x1, x2)

                        output1 = self.classifier(z1).squeeze()
                        output2 = self.classifier(z2).squeeze()
                        loss = (self.criterion(output1, y) + self.criterion(output2, y)) / 2
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            else:
                avg_val_loss = avg_train_loss
                val_losses.append(avg_val_loss)
                print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

            # Save model for current epoch
            if self.save_all_epochs and self.all_epochs_dir:
                epoch_path = os.path.join(self.all_epochs_dir, f"model_epoch_{epoch + 1}.pth")
                torch.save(self.classifier.state_dict(), epoch_path)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.classifier.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.early_stopping and epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Save best model
        if best_model_state:
            self.classifier.load_state_dict(best_model_state)
            torch.save(best_model_state, self.best_model_path)
            print(f"Best model saved to {self.best_model_path}")

        # Plot loss curve
        if self.plot_loss and self.loss_plot_path:
            self._plot_losses(train_losses, val_losses)

        return self.classifier

    def _plot_losses(self, train_losses, val_losses):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train vs. Validation Loss for Classifier on "Trained LSTM Encoder Ouput"')
        plt.legend()
        plt.grid(True)

        details = (
            f'AE Input Size : {self.input_size}\n'
            f'AE Hidden Size: {self.hidden_size}\n'
            f'AE Latent Dim: {self.latent_dim}\n'
            f'AE Num Layers: {self.num_layers}\n'
            f'Batch Size: {self.batch_size}\n'
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.1e}\n'
            f'Sequence Length: {self.seq}\n'

        )
        plt.text(0.5, 0.95, details, transform=plt.gca().transAxes,
                 fontsize=9, va='top', ha='center',
                 bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3', alpha=0.6))

        os.makedirs(os.path.dirname(self.loss_plot_path), exist_ok=True)
        plt.savefig(self.loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
