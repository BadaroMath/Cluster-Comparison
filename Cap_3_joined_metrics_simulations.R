library(tidyverse)
library(clValid)
library(readr)
library(ggplot2)
library("MixSim")
library(latex2exp)
library("plotly")
library(cowplot)
library(dplyr)
library(colorspace)
library(clusterCrit)
options(scipen=999)

operator <- function(BarOmega, K, p, n){
  Q <- MixSim(BarOmega = BarOmega, K = K, p = p, resN = 100000)
  A <- simdataset(n = n, Pi = Q$Pi, Mu = Q$Mu, S = Q$S)
  write.csv(A, paste("simdataset_",
                     BarOmega,"_", K,"_", p,"_", n,".csv", sep=""), row.names=FALSE)
  cat("BarOmega=",BarOmega,"Clusters=",K, "Components=",p, "Obs=",n, "\n")
}
join <- function(BarOmega, K, P, N) {
  cat("--------------------------", 
      "\n", "BarOmega:", 
      BarOmega,"\n","K:", K, "\n", "P:", P)
  cat("\n", "N:", N,"\n")

  DBSCAN <- tryCatch(
    read_csv(paste("F:/TG2/results/resultados_clusters/DBSCAN_", 
                          BarOmega, "_", K, "_", P, "_", N, "_results.csv", 
                          sep=""), show_col_types = FALSE) %>%
      select(Y) %>%
      rename(DBSCAN = Y),
    error = function(e) {
      data.frame(DBSCAN = rep(NA, N))
    }
  )
  
  K_means <- tryCatch(
    read_csv(paste("F:/TG2/results/resultados_clusters/K-means_", 
                   BarOmega, "_", K, "_", P, "_", N, "_results.csv", 
                   sep=""), show_col_types = FALSE) %>%
      select(Y) %>%
      rename(K_means = Y),
    error = function(e) {
      data.frame(K_means = rep(NA, N))
    }
  )
  spectral <- tryCatch(
    read_csv(paste("F:/TG2/results/resultados_clusters/spectral_", 
                   BarOmega, "_", K, "_", P, "_", N, "_results.csv", 
                   sep=""), show_col_types = FALSE) %>%
      select(Y) %>%
      rename(spectral = Y),
    error = function(e) {
      data.frame(spectral = rep(NA, N))
    }
  )
  Gaussian_Mixture <- tryCatch(
    read_csv(paste("F:/TG2/results/resultados_clusters/Gaussian_Mixture_", 
                   BarOmega, "_", K, "_", P, "_", N, "_results.csv", 
                   sep=""), show_col_types = FALSE) %>%
      select(Y) %>%
      rename(Gaussian_Mixture = Y),
    error = function(e) {
      data.frame(Gaussian_Mixture = rep(NA, N))
    }
  )
  Fuzzy_C_means <- tryCatch(
    read_csv(paste("F:/TG2/results/resultados_clusters/Fuzzy_C-means_", 
                   BarOmega, "_", K, "_", P, "_", N, "_results.csv", 
                   sep=""), show_col_types = FALSE) %>%
      mutate(
        Y = apply(select(., starts_with("Y_")), 1,
                        function(x) which.max(x) - 1)
      )  %>%
      select(Y) %>%
      rename(Fuzzy_C_means = Y),
    error = function(e) {
      data.frame(Fuzzy_C_means = rep(NA, N))
    })
    
    dados <- tryCatch(
      read_csv(paste("F:/TG2/results/simulações/simdataset_", 
                     BarOmega,"_", K,"_", P,"_", N,".csv", 
                     sep=""),
               show_col_types = FALSE),
      error = function(e) {
        rerun <- operator(BarOmega, K, P, N)
        read_csv(paste("F:/TG2/results/simulações/simdataset_", 
                       BarOmega,"_", K,"_", P,"_", N,".csv", 
                       sep=""),
                 show_col_types = FALSE)
      })
    
    df <- cbind(dados, DBSCAN, K_means, spectral, Gaussian_Mixture, 
                Fuzzy_C_means)
    write_csv(df, paste("F:/TG2/results/resultados_gerais/joined_", 
                        BarOmega, "_", K, "_", P, "_", N, ".csv", sep=""))
  
}
  


grid_bomg <- expand.grid(BarOmega = seq(0, 0.6, 0.01),
                         K = c(3),
                         P = c(5),
                         N = c(5000))
grid_k <- expand.grid(BarOmega = 0,
                      K = seq(2, 40),
                      P = c(3),
                      N = c(5000))
grid_p <- expand.grid(BarOmega = 0,
                       K = c(3),
                       P = c(seq(2,30, 2), seq(30, 200, 10)),
                       N = c(5000))
grid_n <- expand.grid(BarOmega = 0,
                      K = 3,
                      P = 5,
                      N = c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))
)

merged_grid <- rbind(grid_bomg,grid_p, grid_n)
merged_grid <- as.data.frame(merged_grid)
merged_grid <- distinct(merged_grid, .keep_all = TRUE)

apply(merged_grid, 1, function(x) join(x[1], x[2], x[3], x[4]))
