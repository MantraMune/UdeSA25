import torch
from models import DeepAE, DenoisingAE
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================
# Récupérer les données depuis le notebook
# ============================

data = torch.load("ecg_data.pt")

X_train = data['X_train']
X_val = data['X_val']

data_loaders = torch.load("ecg_dataloaders.pt")

train_loader = data_loaders['train_loader']
val_loader = data_loaders['val_loader']

# ============================
# Entraîner le modèle Deep Auto-Encoder
# ============================

def train_deepAE(train_loader, epochs=40, device='cpu'):
    model = DeepAE(in_features=X_train.shape[1], latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            # Forward
            x_hat = model(batch)

            # Calcul de la perte
            loss = criterion(x_hat, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"Epoch {epoch+1}/{epochs} — Loss = {epoch_loss/n_batches:.4f}")

    return model

# ============================
# Entraîner le modèle variant Denoising Auto-Encoder
# ============================
def train_denoisingAE(train_loader,epochs=40, noise_factor=0.2, device='cpu'):
    model = DenoisingAE(in_features=X_train.shape[1], latent_dim=16, noise_factor=noise_factor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            # Ajouter du bruit
            noisy_batch = model.add_noise(batch, noise_factor)

            # Forward
            x_hat = model(noisy_batch)

            # Calcul de la perte
            loss = criterion(x_hat, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"[Denoising] Epoch {epoch+1}/{epochs} — Loss = {epoch_loss/n_batches:.4f}")

    return model

