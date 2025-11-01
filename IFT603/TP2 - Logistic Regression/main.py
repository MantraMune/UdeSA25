import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

sns.set_theme()

# Partie 1  : Construction du modèle de régression logistique


# Fonctions log-vraisemblance et gradient
def sigmoid(z):
    return 1 / (1 + np.exp(-z)) # En comparant avec les notes, 
                                # z correspond à a -> [-inf, +inf]

def log_likelihood_logistic(w, X, y):
    η = X @ w  # η = X * β, où β = w
    η = np.clip(η, -30, 30) # Pour éviter overflow, puisque η tend vers ±inf
    pi_i = sigmoid(η)
    return np.sum(y * np.log(pi_i) + (1 - y) * np.log(1 - pi_i))

def gradient_logistic(w, X, y):
    η = X @ w
    η = np.clip(η, -30, 30) # Pour éviter overflow, puisque η tend vers ±inf
    pi_i = sigmoid(η)
    return X.T @ (y - pi_i) # X est une matrice n x (d + 1) et 
                            # (y - pi_i) un vecteur n x 1 et on 
                            # veut un gradient de dimension (d + 1) x 1

# Descente de gradient
def logistic_regression(X, y, learning_rate=1e-4, max_iterations=10000, tolerance=1e-6, X_val=None, y_val=None):
    β = np.random.uniform(-0.01, 0.01, X.shape[1]) # w_d ← rand(-0.01, 0.01)
    ll_history_train = [] # Initialisation liste log-vraisemblance pour train
    ll_history_val = [] # Initialisation liste log-vraisemblance pour val

    for iteration in range(max_iterations):
        # Calcul du gradient
        grad = gradient_logistic(β, X, y)

        # Mise à jour des poids
        β += learning_rate * grad

        # Calcul et stockage de la log-vraisemblance
        ll_train = log_likelihood_logistic(β, X, y)
        ll_history_train.append(ll_train)
        if X_val is not None and y_val is not None:
            ll_val = log_likelihood_logistic(β, X_val, y_val)
            ll_history_val.append(ll_val)
        if iteration > 0 and abs(ll_history_train[-1] - ll_history_train[-2]) < tolerance:
            print(f"Convergence atteinte en {iteration} itérations.")
            break
    return β, ll_history_train, ll_history_val

# Partie 2 : Évaluation du modèle


# Métriques d'évaluation -> implémentation manuelle
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred) # Accuracy = Nombre de bonnes prédictions / Nombre total d'observations
                                     # y_true == y_pred retourne un tableau de booléens (True/False)
                                     # np.mean() convertit les booléens en 1/0 et calcule la moyenne

def compute_precision(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1)) # Vrais positifs (on additionne les observations où y_true (attendue) et y_pred (prédite) sont tous deux égaux à 1)
    FP = np.sum((y_true == 0) & (y_pred == 1)) # Faux positifs (on additionne les observations où y_true est 0 (attendue) et y_pred est 1 (prédite))
    return TP / (TP + FP) 

def compute_recall(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1)) # Vrais positifs
    FN = np.sum((y_true == 1) & (y_pred == 0)) # Faux négatifs (on additionne les observations où y_true est 1 (attendue) et y_pred est 0 (prédite))
    return TP / (TP + FN)

def compute_f1(precision, rappel):
    return 2 * ((precision * rappel) / (precision + rappel))

def compute_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1)) # Vrais positifs
    TN = np.sum((y_true == 0) & (y_pred == 0)) # Vrais négatifs
    FP = np.sum((y_true == 0) & (y_pred == 1)) # Faux positifs
    FN = np.sum((y_true == 1) & (y_pred == 0)) # Faux négatifs
    return np.array([[TN, FP],
                     [FN, TP]])

# Visualisations
def plot_log_likelihood(history_train, history_val, y_train, y_val): # Affiche l'évolution de la log-vraisemblance
    plt.figure(figsize=(8, 5))
    plt.plot(np.array(history_train)/len(y_train), label='Training', color='green')
    plt.plot(np.array(history_val)/len(y_val), label='Validation', color='orange')
    plt.xlabel('Itérations')
    plt.ylabel('Log-vraisemblance normalisée')
    plt.title('Évolution de la log-vraisemblance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Affiche la relation entre les probabilités prédites et les classes observées
def plot_predictions_vs_observations(prob_pred, y_val):
    plt.figure(figsize=(8, 5))
    plt.scatter(prob_pred, y_val, alpha=0.6, color='royalblue')
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Seuil de décision (0.5)')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Classe observée')
    plt.title('Régression logistique | Prédictions vs Observations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Affiche une matrice de confusion
def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Classe 0', 'Classe 1'], 
                yticklabels=['Classe 0', 'Classe 1'])
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion')
    plt.tight_layout()
    plt.show()

# Affiche la courbe ROC
def plot_roc_curve(fpr, tpr, auc): # fpr = False Positive Rate, tpr = True Positive Rate, auc = Area Under Curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f})', color="darkorange")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Faux positifs (FPR)')
    plt.ylabel('Vrais positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# PCA (Principal Component Analysis) : Visualisation de la frontière de décision
def manual_pca(X, n_components=2):
    # Étape 1 : Centrer les données
    X_centered = X - np.mean(X, axis=0)

    # Étape 2 : Calculer la matrice de covariance (p x p)
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Étape 3 : Décomposition spectrale (valeurs et vecteurs propres)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix) # eigh() pour matrice symétrique

    # Étape 4 : Tri des composantes principales par ordre décroissant de variance
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    # Étape 5 : Sélection des n premières composantes principales
    components = eigvecs[:, :n_components]

    # Étape 6 : Projection des données centrées sur les composantes principales
    X_proj = X_centered @ components
    return X_proj, components 

# Affiche la frontière de décision dans un plan PCA
def plot_decision_boundary_pca(X, y, β, title="Frontière de décision (PCA manuelle)"):
    # Étape 1 : Réduction de dimennsion en 2D avec PCA manuelle
    X_proj, components = manual_pca(X, n_components=2)
    # Étape 2 : Définition d'une grille couvrant l'espace projeté
    x_min, x_max = X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1
    y_min, y_max = X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), # meshgrid() crée des tuples de coordonnées
                         np.linspace(y_min, y_max, 300)) # linspace() crée des vecteurs espacés linéairement
    grid = np.c_[xx.ravel(), yy.ravel()] # Grille 2D, ravel() réduit un tableau multidimensionnel en 1D
    # Étape 3 : Inversion de la projection par transposition pour revenir à l'espace original
    # Plan PCA 2D -> Espace variables initiales
    X_grid_original = grid @ components.T + np.mean(X, axis=0)
    # Étape 4 : Ajout de l'interception pour l'application du modèle linéaire
    X_grid_augmented = np.column_stack((np.ones(X_grid_original.shape[0]), X_grid_original))
    # Étape 5 : Calcul des probabilités prédites sur la régression logistique
    probs = sigmoid(X_grid_augmented @ β).reshape(xx.shape)
    # Étape 6 : Tracé de la frontière de décision
    plt.figure(figsize=(7, 5))
    # a) Zones colorées de prédiction
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'orange'])
    # b) Ligne de séparation (probabilité = 0.5)
    plt.contour(xx, yy, probs, levels=[0.5], colors='k', linewidths=1)
    # c) Affichage des données projetées (classes 0 et 1)
    plt.scatter(X_proj[y == 0, 0], X_proj[y == 0, 1], label='Classe 0', alpha=0.6, c='blue')
    plt.scatter(X_proj[y == 1, 0], X_proj[y == 1, 1], label='Classe 1', alpha=0.6, c='orange')
    # d) Mise en forme
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Charger les données
    data = load_breast_cancer()
    X_raw, y = data.data, data.target
    feature_names = data.feature_names

    # Division train / validation AVANT standardisation
    X_raw_train, X_raw_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=0.2, random_state=8302
    ) # -> X_train = training set, X_val = testing set, même chose pour y

    # Standardisation des données
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_raw_train)
    X_val_std = scaler.transform(X_raw_val)

    # Ajout de l'interception
    X_train = np.column_stack((np.ones(X_train_std.shape[0]), X_train_std))
    X_val = np.column_stack((np.ones(X_val_std.shape[0]), X_val_std))
    variables = ['Intercept'] + list(feature_names)

    # Entraînement du modèle
    β_hat, history_train, history_val = logistic_regression(X_train, y_train, X_val=X_val, y_val=y_val)

    # Prédictions
    prob_pred = sigmoid(X_val @ β_hat) # Probabilités prédites pour le set de validation
    y_pred_class = (prob_pred >= 0.5).astype(int) # Classe prédite (seuil à 0.5)

    # Affichage des coefficients estimés
    print("Coefficients estimés :")
    for name, coef in zip(['Intercept'] + list(feature_names), β_hat):
        print(f"{name}: {coef:.4f}")

    # Visualisation apprentissage et performance
    plot_log_likelihood(history_train, history_val, y_train, y_val)
    plot_predictions_vs_observations(prob_pred, y_val)

    # Évaluation manuelle
    accuracy = compute_accuracy(y_val, y_pred_class)
    precision = compute_precision(y_val, y_pred_class)
    recall = compute_recall(y_val, y_pred_class)
    f1_score = compute_f1(precision, recall)
    conf_matrix = compute_confusion_matrix(y_val, y_pred_class)
    fpr, tpr, _ = roc_curve(y_val, prob_pred) # Important : usage des fonctions de la librairie sckit-learn pour ROC et AUC
    auc = roc_auc_score(y_val, prob_pred)
    print("\nMétriques d'évaluation :")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Précision: {precision:.3f}")
    print(f"Rappel: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    print(f"AUC: {auc:.3f}")
    
    # Visualisations finales
    plot_confusion_matrix(conf_matrix)
    plot_roc_curve(fpr, tpr, auc)
    plot_decision_boundary_pca(X_train[:, 1:], y_train, β_hat)


# Exécution 
main()

# Interprétation des coefficients estimés : 
# Si on regarde les mesures moyennes de forme, on remarque que pour la plupart des coefficients, on a des valeurs négatives, 
# signifiant que ces mesures sont plus associées à la classe 0 (tumeur bénigne). C'est pour certaines mesures de la régularité des contours, 
# donc la compactivité moyenne des données, la symétrie moyenne des données et la dimension fractale moyenne des données, 
# qu'on retrouve des coefficients positifs, les associant donc à la classe 1 (tumeur maligne).
# Pour les mesures d'erreur, on remarque la même tendance avec les mêmes types de données (cette fois-ci associés à l'erreur et non à la moyenne), donc la conclusion
# est la même : la plupart des mesures sont associées à la classe 0, sauf la compacité, la symétrie et la dimension fractale, qui sont associées à la classe 1.
# Pour les pires valeurs mesurées sur la tumeur, on est surtout sur des coefficients négatifs pour la grande majorité des mesures, sauf pour la compactivité, qui est
# positif. Cela signifie que les pires valeurs de compacité sont associées à la classe 1 (tumeur maligne), tandis que les autres mesures sont associées à la classe 0 (tumeur bénigne).
# Petit rappel, afin de mieux comprendre l'impact des coefficients sur les probabilités de classes, il faut comprendre que, dans notre cas,
# lorsque qu'un coefficient β < 0, la variable associée est corrélée à la classe 0 (tumeur bénigne), et lorsque β > 0, la variable associée est corrélée à la classe 1 (tumeur maligne).
# Cela est dû au fait que la variable cible y est binaire et donc on définit les classes de la manière suivante : y = 0 pour la classe 0 (tumeur bénigne) et y = 1 pour la classe 1 (tumeur maligne). 