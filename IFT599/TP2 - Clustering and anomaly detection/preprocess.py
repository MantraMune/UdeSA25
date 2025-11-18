from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np


# ================================
# 1. Chargement Hi-Seq
# ================================

def load_hiseq(data_path="hiseq_data.csv", labels_path="hiseq_labels.csv"):
    # Charge X
    X = pd.read_csv(data_path)
    
    # Set sample IDs as index
    X = X.set_index("Unnamed: 0")

    # Charge labels
    labels = pd.read_csv(labels_path)
    labels = labels.set_index("Unnamed: 0")["Class"]

    # Fusion X + y
    X["label"] = labels.reindex(X.index)

    return X


# ================================
# 2. Chargement ECG
# ================================

def load_ecg(path="ecg.npz"):
    """
    Le fichier ecg.npz contient une clé 'ecg'
    colonne -1 = label
    autres colonnes = features
    """
    data = np.load(path)
    ecg = data["ecg"]

    X = pd.DataFrame(ecg[:, :-1])
    y = pd.Series(ecg[:, -1], name="label")

    return X, y


# ================================
# 3. Nettoyage
# ================================

def clean(df):
    """
    - Remplace les NaN par la médiane
    - Supprime les colonnes constantes
    """
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))

    # Suppression colonnes constantes
    nunique = df.apply(pd.Series.nunique)
    const_cols = nunique[nunique == 1].index
    df = df.drop(columns=const_cols, errors='ignore')

    return df


# ================================
# 4. Normalisation
# ================================

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ================================
# 5. PCA
# ================================

def reduce_dimension_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


# ================================
# 6. Pipeline Hi-Seq complet
# ================================

def prepare_hiseq():
    df = load_hiseq()

    # Séparation X / y
    X = df.drop(columns=["label"])
    y = df["label"]

    # Nettoyage
    X = clean(X)

    # Normalisation
    X_scaled, scaler = normalize_data(X)

    # PCA pour visualisation (2D)
    X_pca2d, pca2d = reduce_dimension_pca(X_scaled, n_components=2)

    # PCA pour clustering (100 dimensions)
    X_pca100, pca100 = reduce_dimension_pca(X_scaled, n_components=100)

    return {
        "X_raw": X,
        "y": y,
        "X_scaled": X_scaled,
        "X_pca2d": X_pca2d,
        "X_pca100": X_pca100,
        "scaler": scaler,
        "pca2d": pca2d,
        "pca100": pca100
    }


# ================================
# 7. Pipeline ECG complet
# ================================

def prepare_ecg():
    X, y = load_ecg()

    # Nettoyage
    X = clean(X)

    # Normalisation
    X_scaled, scaler = normalize_data(X)

    # PCA pour visualisation (2D)
    X_pca2d, pca2d = reduce_dimension_pca(X_scaled, n_components=2)

    return {
        "X_raw": X,
        "y": y,
        "X_scaled": X_scaled,
        "X_pca2d": X_pca2d,
        "scaler": scaler,
        "pca2d": pca2d
    }
