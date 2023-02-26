
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score, accuracy_score
import pandas as pd
import numpy as np


def kmeans_simul(BarOmega, K, p, n):
    df = pd.read_csv(f"F:/TG2/results/simulações/simdataset_{BarOmega}_{K}_{p}_{n}.csv")
    X = df.drop(columns=['id'])
    Y = df["id"]
    km = cluster.KMeans(
            n_clusters=K,
            n_init = 10,
            max_iter = 100000
        )
    km.fit(X)
    y_pred = km.predict(X).astype(int)
    df_results = pd.DataFrame({'Y': y_pred, 'id': Y})
    df_results.to_csv(f"F:/TG2/results/kmeans_simuls/kmeans_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)




for BarOmega in range(0, 61, 1):
    try:
        kmeans_simul(int(BarOmega/100), 3, 5, 2000)
    except:
        print("Deu erro, mas continuando...")



for K in range(2, 40, 1):
    try:
        kmeans_simul(0, K, 3, 2000)
    except:
        print("Deu erro, mas continuando...")


for n in np.concatenate((np.arange(100, 5100, 100), np.arange(5000, 100100, 10000), np.arange(100000, 2000100, 100000))):
    try:
        kmeans_simul(0, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in np.concatenate((np.arange(2, 21, 2), np.arange(30, 210, 10))):
    try:
        kmeans_simul(0, 3, p, 2000)
    except:
        print("Deu erro, mas continuando...")    

################ General (individual) BarOmega = 0.05  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.05, K, 3, 2000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.05, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
    try:
        kmeans_simul(0.05, 3, p, 2000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.1  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.1, K, 3, 2000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.1, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
    try:
        kmeans_simul(0.1, 3, p, 2000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.15  ##############
for K in range(2, 41):
    try:
        kmeans_simul(0.15, K, 3, 2000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.15, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
    try:
        kmeans_simul(0.15, 3, p, 2000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.20  ##############

for K in range(2, 41):
    try:
        kmeans_simul(0.2, K, 3, 2000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        kmeans_simul(0.2, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
    try:
        kmeans_simul(0.2, 3, p, 2000)
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

for n in range(1000, 10000, 1000):
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            kmeans_simul(0, 3, p, n)
        except:
            print("Deu erro, mas continuando...")