import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


# ============================
# 1. Aperçu et informations
# ============================

def describe(df, name="Dataset"):
    print(f"\n=== {name} ===")
    print("Shape :", df.shape)
    print("Nombre de NaN :", df.isnull().sum().sum())
    print(df.head())


# ============================
# 2. Visualisation PCA 2D
# ============================

def plot_pca_2d(X_2d, labels=None, title="Projection PCA 2D"):
    plt.figure(figsize=(8, 6))

    if labels is None:
        # Pas de labels = pas de couleurs
        plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10)
    else:
        # Si labels sont textuels → conversion automatique
        if labels.dtype == "object":
            labels = labels.astype("category").cat.codes

        scatter = plt.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=labels, cmap="tab10", s=10
        )
        plt.colorbar(scatter)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# ==============================================================================
# Métriques ====================================================================
# ==============================================================================

import time
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

def metrics(labels, data, method, runs=10, **kwargs):
    sil, db, ch, ari, nmi = [], [], [], [], []
    timer, space = [], []

    for seed in range(runs):
        t = time.perf_counter()
        model = method(random_state=seed, **kwargs)
        preds = model.fit_predict(data)
        sil.append(silhouette_score(data, preds))
        db.append(davies_bouldin_score(data, preds))
        ch.append(calinski_harabasz_score(data, preds))
        ari.append(adjusted_rand_score(labels, preds))
        nmi.append(normalized_mutual_info_score(labels, preds))
        t = time.perf_counter() - t
        timer.append(t)


    return {
        "silhouette": (np.mean(sil), np.std(sil)),
        "davies_bouldin": (np.mean(db), np.std(db)),
        "calinski_harabasz": (np.mean(ch), np.std(ch)),
        "adjusted_rand": (np.mean(ari), np.std(ari)),
        "nmi": (np.mean(nmi), np.std(nmi)),
        "time": (np.mean(timer), np.std(timer))
    }

# ============================
# Reconstruction des erreurs pour détection d'anomalies
# ============================

def compute_reconstruction_errors(model, data_loader, device='cpu'):
    model.eval()
    errors = []

    with torch.no_grad():
        for (batch,) in data_loader:
            batch = batch.to(device)
            x_hat = model(batch)
            batch_errors = torch.mean((x_hat - batch) ** 2, dim=1)  # Erreur MSE par échantillon
            errors.extend(batch_errors.cpu().numpy())

    return errors