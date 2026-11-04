"""Temporal attention LSTM autoencoder (new STSM experiment only).

Only the encoder summary changes relative to models.LSTMAutoencoder:
    original: final top-layer hidden state -> Linear(H, LD)
    temporal_attention: all top-layer hidden states -> Linear(H, 1)
        -> softmax over time -> weighted sum -> Linear(H, LD)

Global means that all T encoder outputs contribute to one context vector.
This is scalar temporal pooling, not multi-head self-attention, feature
attention, decoder cross-attention, a VAE, or a classifier.
The original decoder is reused without modifying models.py.
"""
import torch
import torch.nn as nn
from .models import LSTMDecoder

class TemporalAttentionLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(TemporalAttentionLSTMEncoder, self).__init__()
        # Same unidirectional stacked LSTM as the baseline; num_layers=2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, return_attention=False):
        if x.ndim != 3 or x.size(1) == 0 or x.size(2) != self.lstm.input_size:
            raise ValueError('Expected nonempty [batch, time, input_size] input')
        outputs, _ = self.lstm(x)                         # [B,17,H]
        scores = self.temporal_attention(outputs)         # [B,17,1]
        weights = torch.softmax(scores, dim=1)            # normalize over 17 times
        context = (weights * outputs).sum(dim=1)          # [B,H]
        z = self.encoder_fc(context)                      # [B,LD]
        if return_attention:
            return z, weights
        return z


class TemporalAttentionLSTMAutoencoder(nn.Module):
    """One shared AE is applied separately to X_A and X_B.

    Each half-orbit has its own z and attention weights. Nothing is concatenated
    or averaged in latent space. Training averages the two MSE losses instead.
    Default forward returns only decoded [B,T,F], so existing AnomalyDetector
    works unchanged. Opt in to (decoded, weights) for inspection only.
    """
    model_type = 'temporal_attention'

    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(TemporalAttentionLSTMAutoencoder, self).__init__()
        self.encoder = TemporalAttentionLSTMEncoder(
            input_size, hidden_size, num_layers, latent_dim)
        # Unchanged baseline decoder: repeat z T times -> stacked LSTM -> FC H->F.
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, latent_dim)
        self.use_internal_classifier = False

    def forward(self, x, return_attention=False):
        seq_len = x.size(1)
        z, weights = self.encoder(x, return_attention=True)
        decoded = self.decoder(z, seq_len)
        if return_attention:
            return decoded, weights                         # [B,T,F], [B,T,1]
        return decoded


class FeatureTemporalAttentionLSTMEncoder(nn.Module):
    """Feature attention before LSTM; temporal pooling after LSTM."""

    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super().__init__()

        # NEW: one feature score per input feature at each timestep.
        self.feature_attention = nn.Linear(input_size, input_size)

        # Same encoder structure as the temporal-only model.
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )

        self.temporal_attention = nn.Linear(hidden_size, 1)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, return_attention=False):
        if (
            x.ndim != 3
            or x.size(1) == 0
            or x.size(2) != self.lstm.input_size
        ):
            raise ValueError(
                "Expected nonempty [batch, time, input_size] input"
            )

        # Feature attention: normalize over features, independently
        # for each sample and timestep.
        feature_scores = self.feature_attention(x)       # [B,T,F]
        feature_weights = torch.softmax(
            feature_scores, dim=2
        )                                                  # [B,T,F]

        weighted_input = x * feature_weights             # [B,T,F]

        outputs, _ = self.lstm(weighted_input)            # [B,T,H]

        # Temporal attention: normalize over timesteps.
        temporal_scores = self.temporal_attention(outputs)  # [B,T,1]
        temporal_weights = torch.softmax(
            temporal_scores, dim=1
        )                                                  # [B,T,1]

        context = (temporal_weights * outputs).sum(dim=1)  # [B,H]
        z = self.encoder_fc(context)                       # [B,LD]

        if return_attention:
            return z, {
                "feature": feature_weights,
                "temporal": temporal_weights,
            }

        return z


class FeatureTemporalAttentionLSTMAutoencoder(nn.Module):
    """Shared AE for paired inputs; original decoder is reused."""

    model_type = "TF_attention"

    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super().__init__()

        self.encoder = FeatureTemporalAttentionLSTMEncoder(
            input_size,
            hidden_size,
            num_layers,
            latent_dim,
        )

        self.decoder = LSTMDecoder(
            hidden_size,
            input_size,
            num_layers,
            latent_dim,
        )

        self.use_internal_classifier = False

    def forward(self, x, return_attention=False):
        if return_attention:
            z, attention = self.encoder(x, return_attention=True)
            decoded = self.decoder(z, x.size(1))
            return decoded, attention

        z = self.encoder(x)
        return self.decoder(z, x.size(1))