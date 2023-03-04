
import numpy as np
import pandas as pd

def params(BarOmega, K, p, n):
    try:
        pd.read_csv(f"F:/TG2/results/resultados_clusters/DBSCAN_{BarOmega}_{K}_{p}_{n}_results.csv")
        pd.read_csv(f"F:/TG2/results/resultados_clusters/spectral_{BarOmega}_{K}_{p}_{n}_results.csv")
    except:
        with open('F:/TG2/results/resultados_clusters/faltantes_dbscan_spectral.txt', 'a') as f:
            f.write(f"{BarOmega}, {K}, {p}, {n}\n") 
    else:
        None

    try:
        pd.read_csv(f"F:/TG2/results/resultados_clusters/Gaussian_Mixture_{BarOmega}_{K}_{p}_{n}_results.csv")
        pd.read_csv(f"F:/TG2/results/resultados_clusters/K-means_{BarOmega}_{K}_{p}_{n}_results.csv")
        pd.read_csv(f"F:/TG2/results/resultados_clusters/Fuzzy_C-means_{BarOmega}_{K}_{p}_{n}_results.csv")
    except:
        with open('F:/TG2/results/resultados_clusters/faltantes_convex.txt', 'a') as f:
            f.write(f"{BarOmega}, {K}, {p}, {n}\n") 
    else:
        None

for BarOmega in range(0, 61, 1):
    if BarOmega == 0:
        BarOmea = int(0)
    else:
        BarOmega = int(BarOmega)/100
    try:
        params(BarOmega, 3, 5, 5000)
    except:
        print("Deu erro, mas continuando...")
    else:
        None

for K in range(2, 40, 1):
    try:
        params(0, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in np.concatenate((np.arange(100, 5100, 100), np.arange(5000, 100100, 10000), np.arange(100000, 2000100, 100000))):
    try:
        params(0, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")


for p in np.concatenate((np.arange(2, 21, 2), np.arange(30, 210, 10))):
    try:
        params(0, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")    

################ General (individual) BarOmega = 0.05  ##############

for K in range(2, 41):
    try:
        params(0.05, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        params(0.05, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")

for p in list(range(2, 21, 2)) + list(range(30, 71, 10)):
    try:
        params(0.05, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.1  ##############

for K in range(2, 41):
    try:
        params(0.1, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        params(0.1, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 51, 10)):
    try:
        params(0.1, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.15  ##############

for K in range(2, 41):
    try:
        params(0.15, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")


for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        params(0.15, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 41, 10)):
    try:
        params(0.15, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")


################ General (individual) BarOmega = 0.20  ##############


for K in range(2, 41):
    try:
        params(0.2, K, 3, 5000)
    except:
        print("Deu erro, mas continuando...")

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    try:
        params(0.2, 3, 5, n)
    except:
        print("Deu erro, mas continuando...")


for p in list(range(2, 21, 2)) + list(range(30, 31, 10)):
    try:
        params(0.2, 3, p, 5000)
    except:
        print("Deu erro, mas continuando...")

################ Cluster & Components  ##############

for K in [3, 5, 7, 10, 15, 20, 25]:
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            params(0, K, p, 10000)
        except:
            print("Deu erro, mas continuando...")        


                    
################ Obs & Components  ##############

for n in list(range(100, 5001, 100)) + list(range(5000, 100001, 10000)) + list(range(100000, 2000001, 100000)):
    for p in [3, 5, 8, 10, 15, 30, 50]:
        try:
            params(0, 3, p, n)
        except:
            print("Deu erro, mas continuando...")




import pandas as pd

df = pd.read_csv('F:/TG2/results/resultados_clusters/params_prog.txt', sep = ",", header = None, names = ["BarOmega", "K", "p", "n"])

df = df.drop_duplicates()

df2 = pd.read_csv("F:/TG2/results/resultados_clusters/time_results.txt", sep = ",", header = None, names = ["Method", "BarOmega", "K", "p", "n", "time"])

df2.drop(["Method", "time"], axis = 1, inplace = True)
df2 = df2.drop_duplicates()
merged = pd.merge(df2, df, on=['BarOmega', 'K', 'p', 'n'], how='left', indicator=True)

not_in_df2 = merged[merged['_merge'] != 'both']

print(not_in_df2)