import torch
from models import DeepAE, DenoisingAE
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import model_size_mb, compute_reconstruction_errors
import time
from sklearn.ensemble import IsolationForest

# ============================
# Récupérer les données depuis le notebook
# ============================

data = torch.load("ecg_data.pt")

X_train = data['X_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

# Création des DataLoaders
def dataloaders():
    # Construction des loaders pour AE
    train_loader = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test), batch_size=32, shuffle=False)

    # Pour Isolation Forest, on a besoin des labels séparés
    val_data_IF = (X_val, y_val)
    test_data_IF = (X_test, y_test)

    return {
        "train": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "val_IF": val_data_IF,
        "test_IF": test_data_IF
    }

# ============================
# Entraîner le modèle Deep Auto-Encoder
# ============================

def train_deepAE(train_loader, epochs=40, device='cpu'):
    model = DeepAE(in_features=X_train.shape[1], latent_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.perf_counter()
    epoch_losses = []


    for epoch in range(epochs):
        model.train()
        running_loss = 0
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

            running_loss += loss.item()
            n_batches += 1
        
        epoch_loss = running_loss / n_batches
        epoch_losses.append(epoch_loss)

    train_time = time.perf_counter() - start_time
    loaders = dataloaders()

    stats = {
        "recon_error_stats": compute_reconstruction_errors(model, loaders["val_loader"], device=device),
        "train_time": train_time,
        "model_size_mb": model_size_mb(model)
    }

    return model, stats

# ============================
# Entraîner le modèle variant Denoising Auto-Encoder
# ============================
def train_denoisingAE(train_loader,epochs=40, noise_factor=0.2, device='cpu'):
    model = DenoisingAE(in_features=X_train.shape[1], latent_dim=16, noise_factor=noise_factor)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_time = time.perf_counter()
    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        n_batches = 0

        for (batch,) in train_loader:
            batch = batch.to(device)

            # Ajouter du bruit
            noisy_batch = model.add_noise(batch)

            # Forward
            x_hat = model(noisy_batch)

            # Calcul de la perte
            loss = criterion(x_hat, batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
        
        epoch_loss = running_loss / n_batches
        epoch_losses.append(epoch_loss)

    train_time = time.perf_counter() - start_time
    loaders = dataloaders()

    stats = {
        "recon_error_stats": compute_reconstruction_errors(model, loaders["val_loader"], device=device),
        "train_time": train_time,
        "model_size_mb": model_size_mb(model)
    }

    return model, stats 

# ============================
# Entraîner le modèle Isolation Forest
# ============================

def train_isolation_forest(train_loader, device='cpu'):
 
 t0 = time.perf_counter()

 model = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
 model.fit(X_train)

 train_time = time.perf_counter() - t0

 return model, {
     "train_time": train_time
}

