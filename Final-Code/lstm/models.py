# """
# LSTM Autoencoder architecture for sequence reconstruction.
# Includes:
# - LSTMEncoder
# - LSTMDecoder
# - LSTMAutoencoder
# """


# """
# LSTM Autoencoder architecture for sequence reconstruction.
# Includes:
# - LSTMEncoder
# - LSTMDecoder
# - LSTMAutoencoder
# """


import torch
import torch.nn as nn




class LSTMEncoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, seq_len):
        super(LSTMEncoderV2, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.per_timestep_fc = nn.Linear(hidden_size, latent_dim)  # Output dim for each timestep
        # self.final_fc = nn.Linear(seq_len * 2, latent_dim)  # Flattened projection to latent dim

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)
        # print('out',lstm_out.shape)
        # Apply FC to each timestep
        z_t = self.per_timestep_fc(lstm_out)  # (batch_size, seq_len, 2)
        # print('out after fc',z_t.shape)
        # Flatten across time steps
        z = z_t.reshape(z_t.size(0), -1)  # (batch_size, seq_len * 2)
        
        return z
    
class LSTMDecoderV2(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, latent_dim,seq_len):
        super(LSTMDecoderV2, self).__init__()
        self.lstm = nn.LSTM(latent_dim*seq_len, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, x):
        out_p, (h_n, _) = self.lstm(x)
        # print(h_n.shape)
        # print('out',out_p.shape)
        # print("h_n[-1]",h_n[-1].shape)
        z = self.encoder_fc(h_n[-1])  # Last layer's hidden state
        return z
class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, latent_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, seq_len):
        x = x.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out



class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, use_internal_classifier=False):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, latent_dim)

        self.use_internal_classifier = use_internal_classifier
        if use_internal_classifier:
            self.classifier = nn.Sequential(
                            nn.Linear(latent_dim,16),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(16, 16),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(16, 1 )
                            # nn.ReLU(),
                            # nn.Linear(latent_dim*2, latent_dim),
                            # nn.ReLU(),

                            # nn.Linear(latent_dim , 1)
                        )
    
    def forward(self, x):
        seq_len = x.size(1)
        z = self.encoder(x)
        decoded = self.decoder(z, seq_len)

        if self.use_internal_classifier:
            classification = self.classifier(z)
            return decoded, classification  # reconstruction + classification
        else:
            return decoded  # just reconstruction

class LSTMAutoencoderV2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, seq_len,use_internal_classifier=False):
        super(LSTMAutoencoderV2, self).__init__()
        self.encoder = LSTMEncoderV2(input_size, hidden_size, num_layers, latent_dim,seq_len)
        self.decoder = LSTMDecoderV2(hidden_size, input_size, num_layers, latent_dim,seq_len)

        self.use_internal_classifier = use_internal_classifier
        if use_internal_classifier:
            self.classifier = nn.Sequential(
                            nn.Linear(seq_len*latent_dim,16),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(16, 32),
                            # nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.Dropout(0.5),

                            nn.Linear(32, 16),
                            # nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.Dropout(0.5),

                            nn.Linear(16, latent_dim),
                            nn.ReLU(),
                            nn.Dropout(0.7),

                            nn.Linear(latent_dim, 1)  
                        )


    def forward(self, x):
        seq_len = x.size(1)
        z = self.encoder(x)
        decoded = self.decoder(z, seq_len)

        if self.use_internal_classifier:
            classification = self.classifier(z)
            return decoded, classification  # reconstruction + classification
        else:
            return decoded  # just reconstruction


class LatentClassifier(nn.Module):
    """
    Standalone classifier that takes a latent vector and outputs class (0/1).
    """
    def __init__(self, latent_dim,  output_dim=1):
        super(LatentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, output_dim)
             # For binary classification
        )
        
    def forward(self, z):
        return self.classifier(z)