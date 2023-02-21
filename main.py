

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from kneed import KneeLocator
from sklearn import cluster, mixture
from fcmeans import FCM
import pandas as pd
import warnings
from sklearn.metrics import rand_score, accuracy_score, adjusted_rand_score


df = pd.read_csv("J:/simulações/simdataset_0_3_5_2000000.csv")

X = df.loc[:, df.columns != 'id']
X = X.to_numpy()

def eps_elbow(neighbors, X):
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distance_desc = sorted(distances[:,neighbors -1], reverse=True)
    kneedle = KneeLocator(range(1,len(distance_desc)+1),  #x values
                      distance_desc, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="decreasing") #parameter from figure
    
    return kneedle.knee_y

fcm = FCM(
        n_clusters=3,
        m=1
    )
spectral = cluster.SpectralClustering(
        n_clusters=3,
        eigen_solver="arpack",
        affinity="nearest_neighbors",
    )
dbscan = cluster.DBSCAN(
                eps=eps_elbow(6, X),
                min_samples = 6
            )
gmm = mixture.GaussianMixture(
        n_components=3, 
        covariance_type="full"
    )
kmeans = cluster.KMeans(
        n_clusters=3,
        n_init = 10,
        max_iter = 10000
    )

clustering_algorithms = (
    ("K-means", kmeans),
    #("Fuzzy C-means", fcm),
    ("Spectral\nClustering", spectral),
    ("DBSCAN", dbscan),
    ("Gaussian\nMixture", gmm)
)



def pred(X, algorithm):
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

    if hasattr(algorithm, "labels_"):
        y_pred = algorithm.labels_.astype(int)
    else:
        y_pred = algorithm.predict(X)
    return y_pred


for name, algorithm in clustering_algorithms:
    ID = df["id"].to_numpy()
    # catch warnings related to kneighbors_graph
    
    labels = pred(X, algorithm)
    rand = rand_score(ID, labels)
    classprop = accuracy_score(ID, labels)
    vi = adjusted_rand_score(ID, labels)


    print(name,rand, classprop, vi)



from sklearn.neighbors import NearestNeighbors
import pandas as pd
from kneed import KneeLocator
from sklearn import cluster, mixture
from fcmeans import FCM
import pandas as pd
import warnings
from sklearn.metrics import rand_score


df = pd.read_csv("J:/simulações/simdataset_0_3_5_9e+05.csv")

X = df.loc[:, df.columns != 'id']
X = X.to_numpy()



import plotly.express as px

neighbors = 6
# X_embedded is your data
nbrs = NearestNeighbors(n_neighbors=neighbors ).fit(X)
distances, indices = nbrs.kneighbors(X)
distance_desc = sorted(distances[:,neighbors-1], reverse=True)
px.line(x=list(range(1,len(distance_desc )+1)),y= distance_desc )



from kneed import KneeLocator
kneedle = KneeLocator(range(1,len(distance_desc)+1),  #x values
                      distance_desc, # y values
                      S=1.0, #parameter suggested from paper
                      curve="convex", #parameter from figure
                      direction="decreasing") #parameter from figure





















#def dbscan_param(X, ID):
#    df_results = pd.DataFrame(
#        {
#            "nbs": [0],
#            "eps": [0],
#            "ris": [0],
#        }
#    )
#    for nbs in range(3,50, 2):
#        for eps in range(1, 101, 5):
#            dbscan = cluster.DBSCAN(
#                eps=eps/200,
#                min_samples = nbs
#            )
#            ris = pred(X, dbscan)
#            df_results = pd.concat([df_results,
#                pd.DataFrame.from_records([
#                {
#                    "nbs": nbs,
#                    "eps": eps,
#                    "ris": rand_score(ID, ris),
#                }]
#                )])
#
#    df_results = df_results.reset_index()
#    nbs_opt = df_results.loc[df_results['ris'].idxmax()]["nbs"]
#    eps_opt = df_results.loc[df_results['ris'].idxmax()]["eps"]/200
#
#    return(nbs_opt, eps_opt)




# %%
