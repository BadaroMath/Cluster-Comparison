
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from fcmeans import FCM

def clustering_K(BarOmega, K, p, n):

    print(f"Start with BarOmega: {BarOmega}, K: {K}, p: {p}, n: {n}")
    df = pd.read_csv(f"F:/TG2/results/simulações/simdataset_{BarOmega}_{K}_{p}_{n}.csv")
    X = df.drop(columns=['id'])
    X = StandardScaler().fit_transform(X)
    Y = df["id"]


    fcm = FCM(
            n_clusters=K
        )


    gmm = mixture.GaussianMixture(
            n_components=K, 
            covariance_type="full"
        )
    kmeans = cluster.KMeans(
            n_clusters=K,
            n_init = 10,
            max_iter = 10000
        )
    clustering_algorithms = (
        ("K-means", kmeans),
        ("Fuzzy_C-means", fcm),
        ("Gaussian_Mixture", gmm)
    )
    
    for name, algorithm in clustering_algorithms:
        print(f"Method: {name}")

        t0 = time.time()

        algorithm.fit(X)

        t1 = time.time()

        time_taken = t1 - t0

        with open('F:/TG2/results/resultados_clusters/time_results.txt', 'a') as f:
            f.write(f"{name}, {BarOmega}, {K}, {p}, {n}, {time_taken:.4f}\n")        
        if hasattr(algorithm, "labels_"):
            y_pred = algorithm.labels_.astype(int)
            df_results = pd.DataFrame({'Y': y_pred, 'id': Y})
            df_results.to_csv(f"F:/TG2/results/resultados_clusters/{name}_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)
        elif name=="Fuzzy_C-means":
            y_pred = algorithm.u.astype(float)
            df_results = pd.DataFrame(y_pred)
            df_results.columns = ['Y_' + str(col) for col in df_results.columns]
            df_results["id"] = Y
            df_results.to_csv(f"F:/TG2/results/resultados_clusters/{name}_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)
        else:
            y_pred = algorithm.predict(X).astype(int)    
            df_results = pd.DataFrame({'Y': y_pred, 'id': Y})
            df_results.to_csv(f"F:/TG2/results/resultados_clusters/{name}_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)