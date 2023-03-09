
library(readr)
library(ggplot2)
library("MixSim")
library(latex2exp)
library("plotly")
library(cowplot)
library(dplyr)
library(colorspace)
library(clusterCrit)
library(future.apply)

plan(multiprocess, workers = 4)
setwd("F:/TG2")
options(scipen=999)


metrics <- function(BarOmega, K, P, N){
  cat("--------------------------", 
      "\n", "BarOmega:", 
      BarOmega,"\n","K:", K, "\n", "P:", P)
  cat("\n", "N:", N,"\n")  
  dados <- read_csv(paste("F:/TG2/results/resultados_gerais/joined_", 
                          BarOmega, "_", K, "_", P, "_", N, ".csv", sep=""), 
                    show_col_types = FALSE)
  methods <- c("DBSCAN", "spectral", "Gaussian_Mixture", "K_means", "Fuzzy_C_means")
  # Executar operações em paralelo
  future_lapply(methods, function(Method){
    cat("\n", "Method:", Method,"\n") 
    if(length(unique(dados[,Method])) != length(unique(dados[,"id"]))) {
      cat("Partition lengths do not match for", Method, "\n")
      return(NULL)
    }
    if(sum(is.na(dados[[Method]])) > 0) {
      print("A coluna y possui valores NA.")
    }else{
    indexs <- intCriteria(
      as.matrix(select(dados, -id, -DBSCAN, -spectral, 
                       -Gaussian_Mixture, -K_means, -Fuzzy_C_means)),
      as.integer(unlist(dados[,Method])),
      c("Dunn", "Silhouette")
    )
    RI <- RandIndex(dados$id, dados[[Method]])
    cat(toString(
      c(                        
        Method,
        BarOmega,
        K,
        P,
        N,
        round(RI$R, 3),
        round(RI$AR, 3),
        round(ClassProp(dados$id, dados[[Method]]), 3),
        round(VarInf(dados$id, dados[[Method]]), 3),
        round(indexs$dunn, 3),
        round(indexs$silhouette, 3)
      )
    ),
    file="F:/TG2/results/resultados_metricas/total_metrics.txt", 
    append = TRUE, sep="\n")
    }
  })
}


#grid_bomg <- expand.grid(BarOmega = seq(0, 0.6, 0.01),
#                         K = c(3),
#                         P = c(5),
#                         N = c(5000))
#grid_n <- expand.grid(BarOmega = c(0, 0.05, 0.1, 0.15, 0.2),
#                      K = c(3),
#                      P = c(5),
#                      N = c(50000, 1000000))
#grid_k <- expand.grid(BarOmega = c(0, 0.05, 0.1, 0.15, 0.2),
#                      K = c(3, 5, 10, 20, 40),
#                      P = c(5),
#                      N = c(5000))
grid_p1 <- expand.grid(BarOmega = c(0),
                       K = c(3),
                       P = c(seq(2,30, 2), seq(30, 200, 10)),
                       N = c(5000))
#grid_p2 <- expand.grid(BarOmega = c(0.05),
#                       K = c(3),
#                       P = c(seq(2,20, 2), seq(30, 60, 10)),
#                       N = c(5000))
#grid_p3 <- expand.grid(BarOmega = c(0.1),
#                       K = c(3),
#                       P = c(seq(2,20, 2), seq(30, 50, 10)),
#                       N = c(5000))
#grid_p4 <- expand.grid(BarOmega = c(0.15),
#                       K = c(3),
#                       P = c(seq(2,20, 2), seq(30, 40, 10)),
#                       N = c(5000))
#grid_p5 <- expand.grid(BarOmega = c(0.2),
#                       K = c(3),
#                       P = c(seq(2,20, 2), seq(30, 30, 10)),
#                       N = c(5000))
#grid_kp <- expand.grid(BarOmega = c(0),
#                       K = c(3, 5, 7, 10, 15, 20, 25),
#                       P = c(3, 5, 8, 10, 15, 30, 50),
#                       N = c(10000))
#grid_pn <- expand.grid(BarOmega = c(0),
#                       K = c(3),
#                       P = c(3, 5, 8, 10, 15, 30, 50),
#                       N = c(1000, 5000, 50000, 1000000))
#
#
#merged_grid <- rbind(grid_k,grid_n)
#
#
#merged_grid <- as.data.frame(merged_grid)
#merged_grid <- distinct(merged_grid, .keep_all = TRUE)
#
#


grid_n <- expand.grid(BarOmega = 0,
                      K = 3,
                      P = 5,
                      N = c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))
                      )
gridn <- expand.grid(BarOmega = 0,
                     K = 3,
                     P = 5,
                     N = c(seq(1000, 1000000, 1000)))
apply(grid_bomg, 1, function(x) metrics(x[1], x[2], x[3], x[4]))










Method <- "Gaussian_Mixture"
BarOmega <- 0
K <- 3
P <- 5
N <- 1000000
dados <- read_csv(paste("F:/TG2/results/resultados_gerais/joined_", 
                        BarOmega, "_", K, "_", P, "_", N, ".csv", sep=""), 
                  show_col_types = FALSE)
indexs <- intCriteria(
  as.matrix(select(dados, -id, -DBSCAN, -spectral, 
                   -Gaussian_Mixture, -K_means, -Fuzzy_C_means)),
  as.integer(unlist(dados[,Method])),
  c("Silhouette")
)
                  

round(RandIndex(dados$id, dados[[Method]])$R, 3)
round(RandIndex(dados$id, dados[[Method]])$AR, 3)
round(VarInf(dados$id, dados[[Method]]), 3)
round(indexs$dunn, 3)
round(indexs$silhouette, 3)




check <- metrics %>%
  select(BarOmega, K, P, N) %>%
  distinct()

result <- anti_join(merged_grid, check, by = c("BarOmega", "K", "P", "N"))




