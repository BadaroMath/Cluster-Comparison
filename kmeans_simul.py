
from sklearn import cluster, mixture
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from fcmeans import FCM

def kmeans_simul(BarOmega, K, p, n):

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



for BarOmega in range(0, 61, 1):
    if BarOmega == 0:
        BarOmea = int(0)
    else:
        BarOmega = int(BarOmega)/100
    try:
        kmeans_simul(BarOmega, 3, 5, 5000)
    except:
        print("Deu erro, mas continuando...")
    else:
        None



for K in range(2, 40, 1):
    try:
        kmeans_simul(0, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

import numpy as np
for n in np.concatenate((np.arange(100, 5100, 100), np.arange(5000, 100100, 10000), np.arange(100000, 2000100, 100000))):
    try:
        kmeans_simul(0, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in np.concatenate((np.arange(2, 21, 2), np.arange(30, 210, 10))):
    try:
        kmeans_simul(0, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")    

################ General (individual) BarOmega = 0.05  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.05, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.05, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
    try:
        kmeans_simul(0.05, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.1  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.1, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.1, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 51, 10)):
    try:
        kmeans_simul(0.1, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.15  ##############
for K in range(2, 41):
    try:
        kmeans_simul(0.15, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.15, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
    try:
        kmeans_simul(0.15, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.20  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.2, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.2, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 31, 10)):
    try:
        kmeans_simul(0.2, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")

################ Cluster & Components  ##############

for K in [3, 5, 7, 10, 15, 20, 25]:
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            kmeans_simul(0, K, p, 10000)
        except:
            print("Deu erro, mas continuando...")        


                     
################ Obs & Components  ##############

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            kmeans_simul(0, 3, p, n)
        except:
            print("Deu erro, mas continuando...")