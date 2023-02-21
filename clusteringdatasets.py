#%%
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
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

Compound = pd.read_csv("F:/TG2/results/clusteringdatasets/Compound.csv")
Compound = Compound.drop('label', axis=1)
Compound = Compound.values
Aggregation = pd.read_csv("F:/TG2/results/clusteringdatasets/Aggregation.csv")
Aggregation = Aggregation.drop('label', axis=1)
Aggregation = Aggregation.values
pathbased = pd.read_csv("F:/TG2/results/clusteringdatasets/pathbased.csv")
pathbased = pathbased.drop('label', axis=1)
pathbased = pathbased.values
s2 = pd.read_csv("F:/TG2/results/clusteringdatasets/s2.csv")
s2 = s2.drop('labels', axis=1)
s2 = s2.values
flame = pd.read_csv("F:/TG2/results/clusteringdatasets/flame.csv")
flame = flame.drop('label', axis=1)
flame = flame.values
face = pd.read_csv("F:/TG2/results/clusteringdatasets/face.csv")
face = face.values
#%%


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

#%%
datasets = [
    (
        Compound,
        {
            "damping": 0.77,
            "eps": 0.20,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 6,
            "min_samples": 10,
            "n_neighbors": 3,
            "xi": 0.08,
        },
    ),
    (
        Aggregation,
        {
            "damping": 0.75,
            "eps": 0.18,
            "preference": -220,
            "n_clusters": 7,
            "min_samples": 7,
            "n_neighbors": 25,
            "xi": 0.1,
        },
    ),
    (
        pathbased,
        {
            "eps": 0.25,
            "n_neighbors": 3,
            "min_samples": 3,
            "n_clusters": 3
        },
    ),
    (
        s2,
        {
            "eps": 0.15,
            "n_neighbors": 45,
            "min_samples": 20,
            "xi": 0.1,
            "n_clusters": 15,
            "min_cluster_size": 0.2,
        },
    ),
    (
        flame, 
        {
            "eps": 0.285,
            "min_samples": 3,
            "n_neighbors": 7, 
            "xi": 0.1, 
            "n_clusters": 2,
            "min_cluster_size": 0.2
        }
    ),
    (
        face, 
        {
            "n_neighbors": 20,
            "eps": 0.285,
            "min_samples": 1,
            "n_clusters": 5,
        }
    ),
]
#%%
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
                min_samples = params["min_samples"],
                n_jobs = -1,
                p = 3
            )
    clustering_algorithms = (
        ("K-means", kmeans),
        #("Fuzzy_C-means", fcm),
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
        elif name=="Fuzzy_C-means":
            y_pred = algorithm.u.astype(float)
        else:
            y_pred = algorithm.predict(X)

        df_x = pd.DataFrame(X)
        df_x.columns = ['X' + column for column in df_x.columns.astype(str)]
        df = pd.DataFrame(y_pred)
        df.columns = ['Y_' + column for column in df.columns.astype(str)]
        df_final = pd.concat([df_x, df], axis=1)
  
        #file = open(f"F:/resultados_clusters/toydataset_{name}_{i_dataset}_results.csv", 'a')
        #df_final.to_csv(f"F:/resultados_clusters/toydataset_{name}_{i_dataset}_results.csv", index=False)     

        #if name == "Fuzzy C-means":
        #    fuzzy_labels = np.argmax(y_pred, axis=1)
        #    # Calculando o Índice de Rand ajustado para agrupamentos fuzzy
        #    rand = round(adjusted_rand_score(y, fuzzy_labels), 3)
        #    classprop = round(accuracy_score(y, fuzzy_labels), 3)
        #elif i_dataset == 5:
        #    rand = round(rand_score(np.array([0]*y_pred.size), y_pred), 3)
        #    classprop = round(accuracy_score(np.array([0]*y_pred.size), y_pred), 3)
        #else:
        #    rand = round(rand_score(y, y_pred), 3)
        #    classprop = round(accuracy_score(y, y_pred), 3)
        #file = open(f"F:/resultados_clusters/toydataset_metrics.txt", 'a')
        #file.write(f"{i_dataset},{name},{rand},{classprop}\n")
        
        plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)
        colors = np.array(
            list(
                islice(
                    cycle(
                        [
                            "#377eb8",
                            "#ff7f00",
                            "#4daf4a",
                            "#f781bf",
                            "#a65628",
                            "#984ea3",
                            "#999999",
                            "#e41a1c",
                            "#dede00",
                        ]
                    ),
                    int(max(y_pred) + 1),
                )
            )
        )
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(
            0.99,
            0.01,
            ("%.2fs" % (t1 - t0)).lstrip("0"),
            transform=plt.gca().transAxes,
            size=15,
            horizontalalignment="right",
        )
        plot_num += 1
plt.show()
# %%
