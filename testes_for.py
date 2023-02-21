#%%
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.metrics import rand_score, accuracy_score, adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from fcmeans import FCM

#%%
np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============



pathbased = pd.read_csv("F:/clusteringdatasets/pathbased.csv")
y = pathbased['label']  # labels
pathbased = pathbased.drop('label', axis=1)
pathbased = pathbased.values


# ============
# Set up cluster parameters
# ============

plot_num = 1
#%%
default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 10,
    "xi": 0.05,
    "min_cluster_size": 0.1,
}

best_class_prop = 0
for eps in range(1,1000,5):
    for minPts in range(1, 500, 2):     
        datasets = [
            (
                pathbased, 
                {
                    "eps": eps/1000,
                    "min_samples": minPts,
                    "n_neighbors": 7, 
                    "xi": 0.1, 
                    "min_cluster_size": 0.2
                }
            )
        ]

        for i_dataset, (dataset, algo_params) in enumerate(datasets):
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params["quantile"])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params["n_neighbors"], include_self=False
            )
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            fcm = FCM(
                    n_clusters=params["n_clusters"]
                )
            spectral = cluster.SpectralClustering(
                n_clusters=params["n_clusters"],
                eigen_solver="arpack",
                affinity="nearest_neighbors",
                n_neighbors = params["n_neighbors"]
            )

            gmm = mixture.GaussianMixture(
                    n_components=params["n_clusters"], 
                    covariance_type="full"
                )
            kmeans = cluster.KMeans(
                    n_clusters=params["n_clusters"],
                    n_init = 10,
                    max_iter = 10000
                )
            dbscan = cluster.DBSCAN(
                        eps=params["eps"],
                        min_samples = params["min_samples"]
                    )
            clustering_algorithms = (
                #("K-means", kmeans),
                #("Fuzzy_C-means", fcm),
                #("Spectral_Clustering", spectral),
                ("DBSCAN", dbscan),
                #("Gaussian_Mixture", gmm)
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding"
                        + " may not work as expected.",
                        category=UserWarning,
                    )
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, "labels_"):
                    y_pred = algorithm.labels_.astype(int)
                else:
                    y_pred = algorithm.predict(X)

                classprop = round(accuracy_score(y, y_pred), 3)
                if classprop > best_class_prop:
                    best_class_prop = classprop
                    best_eps = eps
                    best_minPts = minPts
                print(best_eps/1000, best_minPts, best_class_prop) 

# %%
