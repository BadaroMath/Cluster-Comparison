# Carrega os pacotes necessários
library(fpc) # para o DBSCAN
library(kernlab) # para o método espectral

# Carrega o conjunto de dados X

sim <- read_csv("F:/TG2/results/simulações/simdataset_0_8_3_2000.csv")

# Defina o valor inicial de eps
eps <- 0.5

# Crie uma lista para armazenar os resultados
dbscan_results <- list()

# Tente diferentes valores de minPts
for (minPts in seq(5, 15, by = 1)) {
  dbscan <- dbscan(sim, eps = eps, MinPts = minPts)
  
  # Armazene os resultados para cada valor de minPts
  dbscan_results[[as.character(minPts)]] <- dbscan
}


# Calcule a pontuação de silhueta média para cada resultado do DBSCAN
silhouette_scores <- lapply(dbscan_results, function(clustering) {
  silhouette(clustering$cluster, dist(sim))
})

# Encontre o valor de minPts que produz a melhor pontuação de silhueta média
best_minPts <- which.max(sapply(silhouette_scores, function(x) mean(x[, 3])))

# Use o valor de minPts para identificar o melhor resultado do DBSCAN
best_dbscan <- dbscan_results[[as.character(best_minPts)]]

# Calcule a pontuação de silhueta média para o melhor resultado do DBSCAN
best_silhouette <- mean(silhouette(best_dbscan$cluster, dist(sim)))

# Imprima os valores de eps e minPts que produziram a melhor pontuação de silhueta média
cat("Melhores parâmetros do DBSCAN:\n")
cat(sprintf("  eps = %f\n", eps))
cat(sprintf("  minPts = %d\n", best_minPts))
cat(sprintf("  pontuação de silhueta média = %f\n", best_silhouette))






# Define o intervalo de valores de eps para o DBSCAN
eps_range <- seq(0.1, 1.0, by = .1)

# Aplica o DBSCAN para cada valor de eps e minPts
dbscan_results <- lapply(eps_range, function(eps_value) {
  
  # Define o intervalo de valores de minPts
  minPts_range <- seq(5, 15, by = 1)
  
  # Aplica o DBSCAN para cada valor de minPts
  minPts_results <- lapply(minPts_range, function(minPts_value) {
    
    # Executa o DBSCAN com os parâmetros eps e minPts definidos
    dbscan <- dbscan(X, eps = eps_value, MinPts = minPts_value)
    
    # Retorna o número de clusters e a pontuação de silhueta média
    return(list(num_clusters = length(unique(dbscan$cluster)) - 1,
                silhouette_score = mean(silhouette(dbscan$cluster, dist(X)))))
  })
  
  # Retorna os resultados do DBSCAN para cada valor de minPts
  return(minPts_results)
})

# Transforma a lista de resultados em um data frame
dbscan_results_df <- do.call(rbind.data.frame, lapply(dbscan_results, function(x) do.call(rbind.data.frame, x)))

# Encontra os valores ótimos de eps e minPts com base na pontuação de silhueta média
optimal_dbscan <- dbscan_results_df[dbscan_results_df$silhouette_score == max(dbscan_results_df$silhouette_score),]

# Imprime os valores ótimos de eps e minPts
print(paste0("Valores ótimos para DBSCAN: eps=", optimal_dbscan$eps, " minPts=", optimal_dbscan$minPts))

# Define o intervalo de valores de k para o método espectral
k_range <- seq(5, 20, by = 1)

# Aplica o método espectral para cada valor de k
spectral_results <- lapply(k_range, function(k_value) {
  
  # Executa o método espectral com o número de eigenvectors e k definidos
  spectral <- specc(X, centers = length(unique(X$y)), nn = k_value)
  
  # Retorna o número de clusters e a pontuação de silhueta média
  return(list(num_clusters = length(unique(spectral$cluster)),
              silhouette_score = mean(silhouette(spectral$cluster, dist(X)))))
})

# Transforma a lista de resultados em um data frame
spectral_results_df <- do.call(rbind.data.frame, spectral_results)

# Encontra os valores ótimos de number of eigenvectors e k com base na pontuação de silhueta média
optimal_spectral <- spectral_results_df[spectral_results_df$silhouette_score == max(spectral_results_df$silhouette_score),]

# Imprime os valores ótimos de number of eigenvectors e k
print(paste0("Valores ótimos para método espectral: number of eigenvectors=", optimal_spectral$dim, " k=", optimal_spectral$nn))
