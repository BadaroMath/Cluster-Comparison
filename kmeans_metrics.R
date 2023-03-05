
library(readr)
library(ggplot2)
library("MixSim")
library(latex2exp)
library("plotly")
library(cowplot)
library(dplyr)
library(colorspace)
library(clusterCrit)

setwd("F:/TG2")


metrics <- function(BarOmega, K, P, N){
  for(Method in c("DBSCAN", "spectral", "Gaussian_Mixture",
                  "K-means", "Fuzzy_C-means")){
    cat("--------------------------", 
        "\n", "Method: ", Method, "\n", "BarOmega: ", 
        BarOmega, "\n", "K: ", K, "\n", "P: ", P, "\n", "N: ", N)
    labeled <- tryCatch(
      read_csv(paste("F:/TG2/results/resultados_clusters/", Method, "_", 
                     BarOmega,"_", K,"_", P,"_", N, "_results",".csv", 
                     sep=""), 
               show_col_types = FALSE),
      error = function(e) {
        return("FALSE")
      }
    ) 
      
    dados <- tryCatch(
      read_csv(paste("F:/TG2/results/simulações/simdataset_", 
                     BarOmega,"_", K,"_", P,"_", N,".csv", 
                     sep=""),
               show_col_types = FALSE),
      error = function(e) {
        return("FALSE")
      }
    )
    if(length(dados) > 1 && length(labeled) > 1){
      if(Method == "Fuzzy_C-means"){
        labeled <- labeled %>%
          mutate(
            Cluster = apply(select(., starts_with("Y_")), 1,
                            function(x) which.max(x) - 1)
                ) %>%
          select(id, Cluster) %>%
          rename(Y = Cluster)
      }
        indexs <- intCriteria(
          as.matrix(dados[, -which(names(dados) == "id")]),
          as.integer(unlist(labeled$Y)), 
          c("Dunn", "Silhouette")
          )
        cat(toString(
                    c(                        
                      Method,
                      BarOmega,
                      K,
                      P,
                      N,
                      round(RandIndex(labeled$id, labeled$Y)$R, 3),
                      round(RandIndex(labeled$id, labeled$Y)$AR, 3),
                      round(ClassProp(labeled$id, labeled$Y), 3),
                      round(VarInf(labeled$id, labeled$Y), 3),
                      round(indexs$dunn, 3),
                      round(indexs$silhouette, 3)
                      )
        ),
        file="F:/TG2/results/resultados_metricas/metrics.txt", 
        append = TRUE, sep="\n")
    }else{return("Arquivos inexistentes.")}
  }
}


################ BarOmega Variation  ##############


for (BarOmega in seq(0, 0.61, 0.01)){
  metrics(BarOmega, 3, 5, 5000)
}


################ Clusters Variation  ##############
for (K in c(20, 40)){
  metrics(0, K, 3, 5000)
}

for (K in c(2, 5, 10, 20, 40)){
  metrics(0.05, K, 3, 5000)
}

for (K in c(2, 5, 10, 20, 40)){
  metrics(0.1, K, 3, 5000)
}

for (K in c(2, 5, 10, 20, 40)){
  metrics(0.15, K, 3, 5000)
}

for (K in c(2, 5, 10, 20, 40)){
  metrics(0.2, K, 3, 5000)
}

################ Dimensions Variation  ##############

for (p in c(seq(2,80, 2), seq(30, 200, 10))){
  metrics(0, 3, p, 5000)
}

for (p in c(seq(2,20, 2), seq(30, 70, 10))){
  metrics(0.05, 3, p, 5000)
}

for (p in c(seq(2,20, 2), seq(30, 50, 10))){
  metrics(0.1, 3, p, 5000)
}

for (p in c(seq(2,20, 2), seq(30, 40, 10))){
  metrics(0.15, 3, p, 5000)
}

for (p in c(seq(2,20, 2), seq(30, 30, 10))){
  metrics(0.2, 3, p, 5000)
}


################ Nrows Variation  ##############

for (n in c(100, 500, 1000, 5000, 50000, 1000000)){
  metrics(0, 3, 5, n)
}

for (n in c(100, 500, 1000, 5000, 50000, 1000000)){
  metrics(0.05, 3, 5, n)
}

for (n in c(100, 500, 1000, 5000, 50000, 1000000)){
  metrics(0.1, 3, 5, n)
}

for (n in c(100, 500, 1000, 5000, 50000, 1000000)){
  metrics(0.15, 3, 5, n)
}

for (n in c(100, 500, 1000, 5000, 50000, 1000000)){
  metrics(0.2, 3, 5, n)
}


################ Cluster & Components  ##############


for (K in c(3, 5, 7, 10, 15, 20, 25)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    metrics(0, K, p, 10000)
  }
}



################ Obs & Components  ##############


for (n in c(1000, 5000, 50000, 1000000)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    metrics(0, 3, p, n)
  }
}




metrics <- read_csv("F:/TG2/results/resultados_metricas/metrics.txt",
                   show_col_types = FALSE, 
                   col_names = c(
                     "Method", 
                     "BarOmega", 
                     "K", 
                     "P", 
                     "N", 
                     "RI", 
                     "AR", 
                     "ClassProp",
                     "VI",
                     "DI"
                     )
                   )

labeled <- tryCatch(
  read_csv(paste("F:/TG2/results/resultados_clusters/K-means_0_3_5_5000_results.csv", 
                 sep=""), 
           show_col_types = FALSE),
  error = function(e) {
    return("FALSE")
  }
) 
dados <- tryCatch(
  read_csv(paste("F:/TG2/results/simulações/simdataset_0_3_5_5000.csv", 
                 sep=""),
           show_col_types = FALSE),
  error = function(e) {
    return("FALSE")
  }
)




