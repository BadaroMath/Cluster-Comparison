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
Compound_label = Compound['label']
Compound = Compound.drop('label', axis=1)
Compound = Compound.values
Aggregation = pd.read_csv("F:/TG2/results/clusteringdatasets/Aggregation.csv")
Aggregation_label = Aggregation['label']
Aggregation = Aggregation.drop('label', axis=1)
Aggregation = Aggregation.values
pathbased = pd.read_csv("F:/TG2/results/clusteringdatasets/pathbased.csv")
pathbased_label = pathbased['label']
pathbased = pathbased.drop('label', axis=1)
pathbased = pathbased.values
s2 = pd.read_csv("F:/TG2/results/clusteringdatasets/s2.csv")
s2_label = s2['labels']
s2 = s2.drop('labels', axis=1)
s2 = s2.values
flame = pd.read_csv("F:/TG2/results/clusteringdatasets/flame.csv")
flame_label = flame['label']
flame = flame.drop('label', axis=1)
flame = flame.values
face = pd.read_csv("F:/TG2/results/clusteringdatasets/face.csv")
#face_label = face['labels']
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
        if i_dataset == 0:
            df_final['labels'] =  Compound_label
        elif i_dataset == 1:
            df_final['labels'] =  Aggregation_label
        elif i_dataset == 2:
            df_final['labels'] =  pathbased_label
        elif i_dataset == 3:
            df_final['labels'] =  s2_label    
        elif i_dataset == 4:
            df_final['labels'] =  flame_label     
        else:
            print("face não tem label")                                        
  
       
        df_final.to_csv(f"F:/TG2/results/clusteringdatasets_results/clustering_{name}_{i_dataset}_results.csv", 
                        index=False)    