from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Spectral Clustering Model
class SpectralClusters:
    def __init__(self, X, K: int = 5, seed: int = 42, sigma: int = 1):
        # ignoring the eigen solver parameters
        clusters = SpectralClustering(n_clusters=K, # 5 cancers mentioned
                                      random_state=seed, # for our 10 run metrics
                                      affinity='rbf', # gaussian kernel similarity
                                      gamma=sigma, # bandwidth: high>1=decrease
                                      assign_labels='kmeans', # end of algorithm
                                      n_init=10, # k-means iterations
                                      random_sate=None, verbose=True).fit(X)
        self.labels = clusters.labels_

if __name__ == "__main__":
    clusters = SpectralClusters()
    print(clusters.labels)

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