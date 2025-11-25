import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys


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
        # Si labels sont textuels alors conversion automatique
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
from memory_profiler import memory_usage

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

def metrics_clustering(labels, data, method, runs=10, **kwargs):
    sil, db, ch, ari, nmi = [], [], [], [], []
    timer, space = [], []

    params = method().get_params(deep=False)
    for seed in range(runs):

        def run_clustering():
            if "random_state" in params:
                model = method(random_state=seed, **kwargs)
            else :
                model = method(**kwargs)
            return model.fit_predict(data)
        
        baseline = memory_usage(max_usage=True)
        t0 = time.perf_counter()
        max_mem, preds = memory_usage(
            (run_clustering, ), 
            max_usage=True,
            retval=True,
            interval=0.01
        )
        t = time.perf_counter() - t0

        timer.append(t)
        space.append(max_mem-baseline)

        sil.append(silhouette_score(data, preds))
        db.append(davies_bouldin_score(data, preds))
        ch.append(calinski_harabasz_score(data, preds))
        ari.append(adjusted_rand_score(labels, preds))
        nmi.append(normalized_mutual_info_score(labels, preds))

    return {
        "silhouette": (np.mean(sil), np.std(sil)),
        "davies_bouldin": (np.mean(db), np.std(db)),
        "calinski_harabasz": (np.mean(ch), np.std(ch)),
        "adjusted_rand": (np.mean(ari), np.std(ari)),
        "nmi": (np.mean(nmi), np.std(nmi)),
        "time": (np.mean(timer), np.std(timer)),
        "space": (np.mean(space), np.std(space))
    }


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def metrics_anomaly(train_function, train_data, val_data, test_data, y_test=None, runs=10, device="cpu"):
   
    accuracies, recalls, precisions, f1s, aucs = [], [], [], [], []
    train_times, mem_usages = [], []

    for seed in range(runs):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Entraîner le modèle
        model, train_stats = train_function(train_data, device=device)
        train_time = train_stats["train_time"]

        # Taille mémoire
        if hasattr(model, "parameters"):
            mem_usage_mb = model_size_mb(model)
        else:
            mem_usage_mb = sys.getsizeof(model) / (1024 * 1024)

        # Détection d'anomalies
        if hasattr(model, "forward") or hasattr(model, "encode"):
            # AE : val_data et test_data sont des DataLoaders
            val_stats = compute_reconstruction_errors(model, val_data, device=device)
            threshold = np.percentile(val_stats["errors"], 25)

            test_stats = compute_reconstruction_errors(model, test_data, device=device)
            errors_test = test_stats["errors"]

            if y_test is None:
                raise ValueError("Pour AE, il faut passer y_test à metrics_anomaly.")
            y_pred = (errors_test > threshold).astype(int)
        else:
            # Isolation Forest : val_data et test_data sont des tuples (X, y)
            X_val, y_val = val_data
            X_test, y_test = test_data

            errors_val = -model.decision_function(X_val)
            threshold = np.percentile(errors_val, 25)

            errors_test = -model.decision_function(X_test)
            y_pred = (errors_test > threshold).astype(int)

        # Calcul des métriques
        accuracies.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, errors_test))

        train_times.append(train_time)
        mem_usages.append(mem_usage_mb)

    return {
        "accuracy": (np.mean(accuracies), np.std(accuracies)),
        "recall": (np.mean(recalls), np.std(recalls)),
        "precision": (np.mean(precisions), np.std(precisions)),
        "f1_score": (np.mean(f1s), np.std(f1s)),
        "roc_auc": (np.mean(aucs), np.std(aucs)),
        "train_time": (np.mean(train_times), np.std(train_times)),
        "mem_usage_mb": (np.mean(mem_usages), np.std(mem_usages))
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

    errors = np.array(errors)
    mean_error = float(errors.mean())
    std_error = float(errors.std())

    return {
        "errors": errors,
        "mean": mean_error,
        "std": std_error
    }

# ==================================================
# Taille mémoire du modèle
# ==================================================

def model_size_mb(model):
    return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # en mégaoctets (float32 = 4 bytes -> 1 byte = 8 bits)

# ==================================================
# Affichage des métriques
# ==================================================

def print_metrics_table(metrics_dict, model_name="Model"):
    """
    Affiche les métriques sous forme de tableau lisible.
    """
    data = {}
    for k, v in metrics_dict.items():
        mean_val, std_val = v
        data[k] = [f"{round(mean_val, 4)} ± {round(std_val, 4)}"]
    
    df = pd.DataFrame(data, index=["Value"])
    print(f"{model_name} metrics:\n")
    print(df)
    print("\n")