import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
