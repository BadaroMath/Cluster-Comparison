import multiprocessing
from kmeans_simul import kmeans_simul
from optimize_params import optimize_parameters
import numpy as np

# cria uma lista de tuplas contendo os argumentos para a função optimize_parameters()
argumentos = [(0, 3, 5, n) for n in list(range(1000, 1000001, 1000))] #, 
lista = []
for BarOmega in range(0, 61, 1):
    if BarOmega == 0:
        BarOmea = int(0)
    else:
        BarOmega = int(BarOmega)/100
    lista.append(BarOmega)
argumentos1 = [(bo, 3, 5, 5000) for bo in lista]
# cria um pool de processos
pool = multiprocessing.Pool(processes=8)

# executa a função em paralelo nos argumentos
resultados = pool.starmap(optimize_parameters, argumentos1)


