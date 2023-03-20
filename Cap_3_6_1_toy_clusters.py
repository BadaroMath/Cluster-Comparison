#%%
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import rand_score, accuracy_score, adjusted_rand_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from fcmeans import FCM

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 1000
#noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
X_small, y_small = datasets.make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.3)
X_large, y_large = datasets.make_circles(n_samples=(250,500), random_state=3, noise=0.04, factor = 0.7)
y_large[y_large==1] = 2
df = pd.DataFrame(np.vstack([X_small,X_large]),columns=['x1','x2'])
df['label'] = np.hstack([y_small,y_large])
df.label.value_counts()
df[["x1", "x2"]].to_numpy()
noisy_circles = (df[["x1", "x2"]].to_numpy(), df["label"].to_numpy())
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8, n_features = 4)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(9 * 2 + 3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)

plot_num = 1

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

datasets = [
    (
        noisy_circles,
        {
            "damping": 0.77,
            "eps": 0.23,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 3,
            "min_samples": 7,
            "n_neighbors": 20,
            "xi": 0.08,
        },
    ),
    (
        noisy_moons,
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "n_neighbors": 7,
            "xi": 0.1,
        },
    ),
    (
        varied,
        {
            "eps": 0.18,
            "n_neighbors": 7,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        aniso,
        {
            "eps": 0.15,
            "n_neighbors": 25,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (blobs, {"min_samples": 7,"n_neighbors": 7, "xi": 0.1, "min_cluster_size": 0.2}),
    (no_structure, {"n_neighbors": 1}),
]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

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
        ("K-means", kmeans),
        ("Fuzzy C-means", fcm),
        ("Spectral_Clustering", spectral),
        ("DBSCAN", dbscan),
        ("Gaussian_Mixture", gmm)
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
        elif name=="Fuzzy C-means":
            y_pred = algorithm.u.astype(float)
        else:
            y_pred = algorithm.predict(X)
        if name == "Fuzzy C-means":
            fuzzy_labels = np.argmax(y_pred, axis=1)
            # Calculando o Índice de Rand ajustado para agrupamentos fuzzy
            rand = round(adjusted_rand_score(y, fuzzy_labels), 3)
            classprop = round(accuracy_score(y, fuzzy_labels), 3)
        elif i_dataset == 5:
            rand = round(rand_score([0]*len(y_pred), y_pred), 3)
            classprop = round(accuracy_score([0]*len(y_pred), y_pred), 3)
        else:
            rand = round(rand_score(y, y_pred), 3)
            classprop = round(accuracy_score(y, y_pred), 3)
        file = open(f"J:/resultados_clusters/toydataset_metrics.txt", 'a')
        file.write(f"{i_dataset},{name},{rand},{classprop}\n")
        file.close()