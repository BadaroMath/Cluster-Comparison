from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import time
import warnings

# Carrega o conjunto de dados X e seus rótulos verdadeiros Y


def optimize_parameters(BarOmega, K, p, n):

    print(f"Start with BarOmega: {BarOmega}, K: {K}, p: {p}, n: {n}")
    path = f"F:/TG2/results/simulações/simdataset_{BarOmega}_{K}_{p}_{n}.csv"
    X = pd.read_csv(path).drop(columns=["id"]).values
    Y = pd.read_csv(path)["id"].values
    # Configurações dos parâmetros do DBSCAN

    eps_values = np.arange(0.05, 3, 0.05).tolist()
    minPts_values = [1] + np.arange(2, 40, 2).tolist()
    #nn_values = [4, 5, 6]
    

    # Realiza a busca em grade para o DBSCAN

    best_dbscan_params = {'eps': None, 'minPts': None, 'score': -1}
    for eps in eps_values:
        for minPts in minPts_values:
            dbscan = DBSCAN(eps=eps, min_samples=minPts)
            t0_db = time.time()
            dbscan_labels = dbscan.fit_predict(X).astype(int) 
            t1_db = time.time()
            dbscan_score = adjusted_rand_score(Y, dbscan_labels)
            print(BarOmega, K, p, n, f"eps: {eps}, minPts: {minPts}, score: {dbscan_score}")
            if dbscan_score > best_dbscan_params['score']:
                best_dbscan_params['eps'] = eps
                best_dbscan_params['minPts'] = minPts
                best_dbscan_params['score'] = dbscan_score
                time_taken_db = t1_db - t0_db
                best_labels = dbscan_labels
            if dbscan_score > 0.95:
                break
        if dbscan_score > 0.9:
            break
    best_dbscan_params['score'] = adjusted_rand_score(Y, best_labels)
    
    with open('F:/TG2/results/resultados_clusters/time_results.txt', 'a') as f:
        f.write(f"DBSCAN, {BarOmega}, {K}, {p}, {n}, {time_taken_db:.4f}\n")     
    df_results = pd.DataFrame({'Y': best_labels, 'id': Y})
    df_results.to_csv(f"F:/TG2/results/resultados_clusters/DBSCAN_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)


    # Realiza a busca em grade para o Spectral Clustering

    #best_sc_params = {'nn': None, 'score': -1}
    #for nn in nn_values:
    #    sc = SpectralClustering(n_clusters=K, 
    #                            affinity='nearest_neighbors', 
    #                            n_neighbors=nn
    #                            )
    #            # catch warnings related to kneighbors_graph
    #    with warnings.catch_warnings():
    #        warnings.filterwarnings(
    #            "ignore",
    #            message="the number of connected components of the "
    #            + "connectivity matrix is [0-9]{1,2}"
    #            + " > 1. Completing it to avoid stopping the tree early.",
    #            category=UserWarning,
    #        )
    #        warnings.filterwarnings(
    #            "ignore",
    #            message="Graph is not fully connected, spectral embedding"
    #            + " may not work as expected.",
    #            category=UserWarning,
    #        )
    #        t0 = time.time()
    #        sc_labels = sc.fit_predict(X).astype(int) 
    #        t1 = time.time()
    #    spec_score = adjusted_rand_score(Y, sc_labels)
    #    if spec_score > best_sc_params['score']:
    #        best_sc_params['nn'] = nn
    #        best_sc_params['score'] = spec_score
    #        time_taken = t1 - t0
    #        best_label_sc = sc_labels
    #    print(f"nn: {nn}, score: {spec_score}")
    #    if spec_score > 0.9:
    #        break
##
    #
    #best_sc_params['score'] = adjusted_rand_score(Y, best_label_sc)
    #with open('F:/TG2/results/resultados_clusters/time_results.txt', 'a') as f:
    #    f.write(f"Spectral, {BarOmega}, {K}, {p}, {n}, {time_taken:.4f}\n") 
    #     
    #df_spec = pd.DataFrame({'Y': best_label_sc, 'id': Y})
    #df_spec.to_csv(f"F:/TG2/results/resultados_clusters/spectral_{BarOmega}_{K}_{p}_{n}_results.csv", index=False)
##
    best_sc_params = {'nn': None, 'score': None}
    with open('F:/TG2/results/resultados_clusters/dbscan_spectral_params.txt', 'a') as f:
            f.write(f'"{path}", {round(best_dbscan_params["eps"], 2)}, {round(best_dbscan_params["minPts"], 2)},{round(best_dbscan_params["score"], 2)}, {best_sc_params["nn"], 2}, {best_sc_params["score"], 2}\n')
#
    print(f"Best DBSCAN parameters: eps={best_dbscan_params['eps']}, MinPts={best_dbscan_params['minPts']} with score: {best_dbscan_params['score']}")
    #print(f"Best Spectral Clustering parameter with score: {best_sc_params['score']}")




#for BarOmega in range(0, 61, 1):
#    if BarOmega == 0:
#        BarOmea = int(0)
#    else:
#        BarOmega = int(BarOmega)/100
#    try:
#        optimize_parameters(BarOmega, 3, 5, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#    else:
#        None
#
#for K in range(2, 40, 1):
#    try:
#        optimize_parameters(0, K, 3, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
#    try:
#        optimize_parameters(0, 3, 5, n)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for p in np.concatenate((np.arange(2, 21, 2), np.arange(30, 210, 10))):
#    try:
#        optimize_parameters(0, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")    
#
################# General (individual) BarOmega = 0.05  ##############
#
#for K in range(16, 41):
#    try:
#        optimize_parameters(0.05, K, 3, 5000, high_overlap = True)
#    except:
#        print("Deu erro, mas continuando...")
#
#for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
#    try:
#        optimize_parameters(0.05, 3, 5, n, default_param = True)
#    except:
#        print("Deu erro, mas continuando...")
#
#for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
#    try:
#        optimize_parameters(0.05, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
################# General (individual) BarOmega = 0.1  ##############
#
#for K in range(2, 41):
#    try:
#        optimize_parameters(0.1, K, 3, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
#    try:
#        optimize_parameters(0.1, 3, 5, n, default_param = True)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for p in list(range(2, 21, 2)) + list(range(30, 51, 10)):
#    try:
#        optimize_parameters(0.1, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
################# General (individual) BarOmega = 0.15  ##############
#
#for K in range(2, 41):
#    try:
#        optimize_parameters(0.15, K, 3, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
#    try:
#        optimize_parameters(0.15, 3, 5, n, default_param = True)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
#    try:
#        optimize_parameters(0.15, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#
################# General (individual) BarOmega = 0.20  ##############
#
#
#for K in range(2, 41):
#    try:
#        optimize_parameters(0.2, K, 3, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
#for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
#    try:
#        optimize_parameters(0.2, 3, 5, n, default_param = True)
#    except:
#        print("Deu erro, mas continuando...")
#
#
#for p in list(range(2, 21, 2)) + list(range(30, 31, 10)):
#    try:
#        optimize_parameters(0.2, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")
#
################# Cluster & Components  ##############
#
#for K in [3, 5, 7, 10, 15, 20, 25]:
#    for p in [3, 5, 8, 10, 15, 30, 50, 100]:
#        try:
#            optimize_parameters(0, K, p, 10000)
#        except:
#            print("Deu erro, mas continuando...")        
#
#
#                        
#    ################ Obs & Components  ##############
#
#for n in [1000, 5000, 50000, 1000000]:
#    for p in [3, 5, 8, 10, 15, 30, 50]:
#        try:
#            optimize_parameters(0, 3, p, n)
#        except:
#            print("Deu erro, mas continuando...")
#
#
#
#
#
#
#
#
#
#for n in [1000, 5000, 50000, 1000000]:
#    for p in [3, 5, 8, 10, 15, 30, 50]:
#        try:
#            optimize_parameters(0, 3, p, n)
#        except:
#            print("Deu erro, mas continuando...")    
#
#
#for p in [60,70]:
#    try:
#        optimize_parameters(0.05, 3, p, 5000)
#    except:
#        print("Deu erro, mas continuando...")    
#
#
#
#
#
#
#optimize_parameters(0.05, 3, 5, 50000)
#optimize_parameters(0.05, 3, 5, 1000000)
#optimize_parameters(0.05, 3, 70, 5000, high_dimensions = True)
#optimize_parameters(0.1, 3, 5, 50000)
#optimize_parameters(0.1, 3, 5, 1000000)
#optimize_parameters(0.1, 3, 50, 5000, high_dimensions = True)
#optimize_parameters(0.15, 3, 5, 50000)
#optimize_parameters(0.15, 3, 5, 1000000)
#optimize_parameters(0.2, 3, 5, 50000)
#optimize_parameters(0.2, 3, 5, 1000000)
#optimize_parameters(0, 25, 3, 10000)
#optimize_parameters(0, 3, 8, 50000)
#optimize_parameters(0, 3, 10, 50000, high_dimensions = True)
#optimize_parameters(0, 3, 15, 50000, high_dimensions = True)
#optimize_parameters(0, 3, 30, 50000, high_dimensions = True)
#optimize_parameters(0, 3, 50, 50000, high_dimensions = True)
#optimize_parameters(0, 3, 8, 1000000)
#optimize_parameters(0, 3, 10, 1000000, high_dimensions = True)
#optimize_parameters(0, 3, 15, 1000000, high_dimensions = True)
#optimize_parameters(0, 3, 30, 1000000, high_dimensions = True)
#optimize_parameters(0, 3, 50, 1000000, high_dimensions = True)
#
#