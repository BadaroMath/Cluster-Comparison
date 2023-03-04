from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import time
import threading

# Carrega o conjunto de dados X e seus rótulos verdadeiros Y


def optimize_parameters(BarOmega, K, p, n, default_param = False, high_overlap = False):

    print(f"Start with BarOmega: {BarOmega}, K: {K}, p: {p}, n: {n}")
    path = f"F:/TG2/results/simulações/simdataset_{BarOmega}_{K}_{p}_{n}.csv"
    X = pd.read_csv(path).drop(columns=["id"]).values
    Y = pd.read_csv(path)["id"].values
    # Configurações dos parâmetros do DBSCAN
    if high_overlap is True:
        eps_values = np.arange(0.2, 0.65, 0.05)
        minPts_values = [1] + np.arange(2, 50, 2).tolist()
    else:
        eps_values = np.arange(0.1, 2.5, 0.05)
        minPts_values = [1] + np.arange(2, 32, 2).tolist()
    nn_values = np.arange(2, 32, 2)

    # Realiza a busca em grade para o DBSCAN
    if default_param is False:
        best_dbscan_params = {'eps': None, 'minPts': None, 'score': -1}
        for eps in eps_values:
            for minPts in minPts_values:
                dbscan = DBSCAN(eps=eps, min_samples=minPts)
                dbscan_labels = dbscan.fit_predict(X)
                dbscan_score = adjusted_rand_score(Y, dbscan_labels)
                #print(f"eps: {eps}, minPts: {minPts}, score: {dbscan_score}")
                if dbscan_score > best_dbscan_params['score']:
                    best_dbscan_params['eps'] = eps
                    best_dbscan_params['minPts'] = minPts
                    best_dbscan_params['score'] = dbscan_score
                if dbscan_score > 0.95:
                    break
            if dbscan_score > 0.9:
                break
    else:
        best_dbscan_params = {'eps': 0.2, 'minPts': 1, 'score': -1}
    dbscan = DBSCAN(eps=best_dbscan_params['eps'], min_samples=best_dbscan_params['minPts'])
    t0_db = time.time()
    y_dbscan = dbscan.fit_predict(X).astype(int)  
    best_dbscan_params['score'] = adjusted_rand_score(Y, y_dbscan)
    t1_db = time.time()
    time_taken_db = t1_db - t0_db
    with open('F:/TG2/results/resultados_clusters/time_results.txt', 'a') as f:
        f.write(f"DBSCAN, {BarOmega}, {K}, {p}, {n}, {time_taken_db:.4f}\n")     
    df_results = pd.DataFrame({'Y': y_dbscan, 'id': Y})
    df_results.to_csv(f"F:/TG2/results/resultados_clusters/DBSCAN_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)

    # Realiza a busca em grade para o Spectral Clustering
    if default_param is False:
        best_sc_params = {'nn': None, 'score': -1}
        for nn in nn_values:
            sc = SpectralClustering(n_clusters=K, affinity='nearest_neighbors', n_neighbors=nn)
            sc_labels = sc.fit_predict(X)
            spec_score = adjusted_rand_score(Y, sc_labels)
            if spec_score > best_sc_params['score']:
                best_sc_params['nn'] = nn
                best_sc_params['score'] = spec_score
            #print(f"nn: {nn}, score: {spec_score}")
            if spec_score > 0.9:
                break
    else:    
        best_sc_params = {'nn': 4, 'score': -1}
    spec = SpectralClustering(n_clusters=K, affinity='nearest_neighbors', n_neighbors=best_sc_params['nn'])
    t0 = time.time()
    y_spec = spec.fit_predict(X).astype(int) 
    t1 = time.time()
    time_taken = t1 - t0
    best_sc_params['score'] = adjusted_rand_score(Y, y_spec)
    with open('F:/TG2/results/resultados_clusters/time_results.txt', 'a') as f:
        f.write(f"Spectral, {BarOmega}, {K}, {p}, {n}, {time_taken:.4f}\n") 
          
    df_spec = pd.DataFrame({'Y': y_spec, 'id': Y})
    df_spec.to_csv(f"F:/TG2/results/resultados_clusters/spectral_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)

    with open('F:/TG2/results/resultados_clusters/dbscan_spectral_params.txt', 'a') as f:
            f.write(f'"{path}", {round(best_dbscan_params["eps"], 2)}, {round(best_dbscan_params["minPts"], 2)},{round(best_dbscan_params["score"], 2)}, {round(best_sc_params["nn"], 2)}, {round(best_sc_params["score"], 2)}\n')

    print(f"Best DBSCAN parameters with score: {best_dbscan_params['score']}")
    print(f"Best Spectral Clustering parameter with score: {best_sc_params['score']}")


for BarOmega in range(0, 61, 1):
    if BarOmega == 0:
        BarOmea = int(0)
    else:
        BarOmega = int(BarOmega)/100
    try:
        optimize_parameters(BarOmega, 3, 5, 5000, high_overlap = True)
    except:
        print("Deu erro, mas continuando...")
    else:
        None

for K in range(2, 40, 1):
    try:
        optimize_parameters(0, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in np.concatenate((np.arange(100, 5100, 100), np.arange(5000, 100100, 10000), np.arange(100000, 2000100, 100000))):
    try:
        optimize_parameters(0, 3, 5, n, default_param = True)
    except:
        print("Deu erro, mas continuando...")


for p in np.concatenate((np.arange(2, 21, 2), np.arange(30, 210, 10))):
    try:
        optimize_parameters(0, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")    

################ General (individual) BarOmega = 0.05  ##############

for K in range(16, 41):
    try:
        optimize_parameters(0.05, K, 3, 5000, high_overlap = True)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        optimize_parameters(0.05, 3, 5, n, default_param = True)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
    try:
        optimize_parameters(0.05, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.1  ##############

for K in range(2, 41):
    try:
        optimize_parameters(0.1, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        optimize_parameters(0.1, 3, 5, n, default_param = True)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 51, 10)):
    try:
        optimize_parameters(0.1, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.15  ##############

for K in range(2, 41):
    try:
        optimize_parameters(0.15, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        optimize_parameters(0.15, 3, 5, n, default_param = True)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
    try:
        optimize_parameters(0.15, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.20  ##############


for K in range(2, 41):
    try:
        optimize_parameters(0.2, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        optimize_parameters(0.2, 3, 5, n, default_param = True)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 31, 10)):
    try:
        optimize_parameters(0.2, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")

################ Cluster & Components  ##############

for K in [3, 5, 7, 10, 15, 20, 25]:
    for p in [3, 5, 8, 10, 15, 30, 50, 100]:
        try:
            optimize_parameters(0, K, p, 10000)
        except:
            print("Deu erro, mas continuando...")        


                        
    ################ Obs & Components  ##############

for n in [1000, 5000, 50000, 1000000, 5000000]:
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            optimize_parameters(0, 3, p, n)
        except:
            print("Deu erro, mas continuando...")





def falt1():
    for n in range(200000, 2100000, 100000):
        try:
            optimize_parameters(0, 3, 5, n, default_param = True)
        except:
            print("Deu erro, mas continuando...")

def falt2():
    for n in range(200000, 2100000, 100000):
        try:
            optimize_parameters(0.05, 3, 5, n, default_param = True)
        except:
            print("Deu erro, mas continuando...")

def falt3():
    for k in range(2, 41, 1):
        try:
            optimize_parameters(0.1, k, 3, 5000)
        except:
            print("Deu erro, mas continuando...")

def falt4():
    for n in range(200000, 2100000, 100000):
        try:
            optimize_parameters(0.1, 3, 5, n)
        except:
            print("Deu erro, mas continuando...")

def falt5():
    for k in range(17, 41, 1):
        try:
            optimize_parameters(0.15, k, 3, 5000)
        except:
            print("Deu erro, mas continuando...")                        

def falt6():
    for k in range(17, 41, 1):
        try:
            optimize_parameters(0.2, k, 3, 5000)
        except:
            print("Deu erro, mas continuando...")             

def falt7():
    for n in range(200000, 2100000, 100000):
        try:
            optimize_parameters(0.2, 3, 5, n)
        except:
            print("Deu erro, mas continuando...")    


def falt8():
    for n in [1000, 5000, 50000, 1000000, 5000000]:
        for p in [3, 5, 8, 10, 15, 30, 50]:
            try:
                optimize_parameters(0, 3, p, n)
            except:
                print("Deu erro, mas continuando...")    

t1 = threading.Thread(target=falt1)
t2 = threading.Thread(target=falt2)
t3 = threading.Thread(target=falt3)
t4 = threading.Thread(target=falt4)
t5 = threading.Thread(target=falt5)
t6 = threading.Thread(target=falt6)
t7 = threading.Thread(target=falt7) 
t8 = threading.Thread(target=falt8)


t1.start()
t2.start()
t3.start()
t4.start()
t5.start()
t6.start()
t7.start()
t8.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()
t6.join()
t7.join()
t8.join()




