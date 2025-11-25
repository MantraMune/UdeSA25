import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

#  Deep Auto-Encoder
class DeepAE(nn.Module):
    def __init__(self, in_features, latent_dim=8):
        super().__init__()
        self.name = "DeepAE"

        # Encodeur
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)   # Bottleneck ou bouchon d'étranglement
        )

        # Décodeur (encodeur mais dans l'autre sens)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_features)
        )

    def forward(self, x): # Prédire la prochaine entrée à partir des entrées précédentes
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
# Denoising Auto-Encoder
class DenoisingAE(DeepAE):
    def __init__(self, in_features, latent_dim=16, noise_factor=0.05):
        super().__init__(in_features=in_features, latent_dim=latent_dim)
        self.name = "DenoisingAE"
        self.noise_factor = noise_factor

        self.encoder = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)   # Bottleneck ou bouchon d'étranglement
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, in_features)
        )
    
    def add_noise(self, x):
        noise = self.noise_factor * torch.randn_like(x)
        return x + noise

    def forward(self, x):
        noisy_x = self.add_noise(x)
        z = self.encoder(noisy_x)
        return self.decoder(z)