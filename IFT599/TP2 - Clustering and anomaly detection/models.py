from sklearn.cluster import SpectralClustering

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