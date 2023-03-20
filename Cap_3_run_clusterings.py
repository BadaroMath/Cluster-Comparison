import multiprocessing
from Cap_3__partitioning_clusters import clustering_K
from Cap_3_density_clusters import search_density
import numpy as np


# grid_bomg
grid_bomg = [(bomg, 3, 5, 5000) for bomg in np.arange(0, 0.61, 0.01)]

# grid_k
grid_k = [(0, k, 3, 5000) for k in range(2, 41)]

# grid_p
grid_p = [(0, 3, p, 5000) for p in np.concatenate((np.arange(2, 31, 2), 
                                                   np.arange(30, 201, 10)))]

# grid_n
grid_n = [(0, 3, 5, n) for n in np.concatenate((np.arange(100, 5001, 100), 
                                                np.arange(5000, 100001, 10000), np.arange(100000, 2000001, 100000)))]


pool = multiprocessing.Pool(processes=8)

resultados = pool.starmap(search_density, grid_bomg)
resultados = pool.starmap(clustering_K, grid_bomg)
resultados = pool.starmap(search_density, grid_k)
resultados = pool.starmap(clustering_K, grid_k)
resultados = pool.starmap(search_density, grid_p)
resultados = pool.starmap(clustering_K, grid_p)
resultados = pool.starmap(search_density, grid_n)
resultados = pool.starmap(clustering_K, grid_n)

