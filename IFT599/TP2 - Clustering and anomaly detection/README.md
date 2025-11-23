```bash
fold -s -w 80 README.md > README.tmp && mv README.tmp README.md
```

# Clustering et Détection d'Anomalies

Groupe constitué de:
- Pathy Jane Noutcha Ngounou
- Luc Nathan Ramasamy
- Abdoul Malick Oumarou Allassane
- Owen Couturier


## Préparation des données
Dans le fichier `preprocess.py` les données des deux datasets "hiseq" et "ecg" 
sont traitées et transformées pour servir respectivement aux parties de 
clustering et de détection d'anomalies. Une visualisation par ACP 2D est 
proposées dans `utils.py` pour visualiser ces données de grandes dimensions.


## Clustering

Pour chaque méthode de clustering "classique", il faudra comparer:
1. une application sur l'ensemble des attributs, 
2. une application sur 100 attributs obtenus par ACP.

Pour chaque application, il faudra évaluer les performances sur les métriques 
statistiques citées ci-après et sur les métriques machines en temps et en 
mémoire.

#### Métriques internes
- Coefficient de silouhette
- Indice de Davies-Bouldin
- Indice de Calinski-Harabasz
#### Métriques externes
- Indice de Rand ajusté
- Information mutuelle normalisée

De plus, chaque métrique sera évaluée sur 10 exécutions initialisées 
aléatoirement afin d'assurer leur indépendance statistique et de mesurer les 
valeurs moyennes des métriques avec un écart-type.

Dans le cas du dataset HiSeq, il y a 5 classes: BRCA (sein), KIRC (rein), COAD 
(côlon), LUAD (poumon) et PRAD (prostate)


### K-NN

### DBSCAN

### Spectral

On va construire une matrice de similarité par noyau gaussien puis calculer le 
Laplacien pour faire un clustering sur son spectre. Cette méthode de 
clustering spectral sert à obtenir des vecteurs propres (les plus petits) sur 
lesquels les données seront projetées de façon à minimiser la coupe du 
graphe pour un clustering k-means classique.

Dans notre cas, le clustering spectral de scikit-learn utilise son module de 
"SpectralEmbedding" pour la partie projection. Celui-ci renvoie un warning sur 
le nombre de composantes connexes:

```bash
sklearn/manifold/_spectral_embedding.py:328: UserWarning: 
Graph is not fully connected, spectral embedding may not work as expected.
```

Il y a plusieurs composantes connexes, plusieurs 0 dans le spectre du 
Laplacien, ce qui peut poser des problèmes sur les clusters comme signalé. 
Ceci est contrôlé par le paramètre gamma, il faut donc reconsidérer la 
matrice de similarité RBF avec un gamma unitaire par défaut:

"Abstract: In kernel methods, the median heuristic has been widely used as a 
way of setting the bandwidth of RBF kernels."

Large sample analysis of the median heuristic.
Damien Garreau - Wittawat Jitkrittum - Motonobu Kanagawa 
Max Planck Institute for Intelligent Systems (2018)
https://arxiv.org/abs/1707.07269