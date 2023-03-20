library(readr)
library(ggplot2)
library("MixSim")
library(latex2exp)
library("fclust")
library("plotly")
library(cowplot)
library(dplyr)
library(colorspace)
library(dbscan)
setwd("F:/TG2/results/clusteringdatasets_results")


spectral_clustering <- function(X, # matrix of data points
                                nn, # the k nearest neighbors to consider
                                n_eig)
  # m number of eignenvectors to keep
{
  kn = nn
  mutual_knn_graph <- function(X, kn)
  {
    D <-
      as.matrix(dist(X)) # matrix of euclidean distances between data points in X
    
    # intialize the knn matrix
    knn_mat <- matrix(0,
                      nrow = nrow(X),
                      ncol = nrow(X))
    
    # find the 10 nearest neighbors for each point
    for (i in 1:nrow(X)) {
      neighbor_index <- order(D[i, ])[2:(nn + 1)]
      knn_mat[i, ][neighbor_index] <- 1
    }
    
    # Now we note that i,j are neighbors iff K[i,j] = 1 or K[j,i] = 1
    knn_mat <- knn_mat + t(knn_mat) # find mutual knn
    
    knn_mat[knn_mat == 2] = 1
    
    return(knn_mat)
  }
  
  graph_laplacian <- function(W, normalized = TRUE)
  {
    stopifnot(nrow(W) == ncol(W))
    
    g = colSums(W) # degrees of vertices
    n = nrow(W)
    
    if (normalized)
    {
      D_half = diag(1 / sqrt(g))
      return(diag(n) - D_half %*% W %*% D_half)
    }
    else
    {
      return(diag(g) - W)
    }
  }
  
  W = mutual_knn_graph(X) # 1. matrix of similarities
  L = graph_laplacian(W) # 2. compute graph laplacian
  ei = eigen(L, symmetric = TRUE) # 3. Compute the eigenvectors and values of L
  n = nrow(L)
  return(ei$vectors[, (n - n_eig):(n - 1)]) # return the eigenvectors of the n_eig smallest eigenvalues
  
}




compound_K_means <- read_csv("clustering_K-means_0_results.csv")
compound_Fuzzy_C_means <-
  read_csv("clustering_Fuzzy_C-means_0_results.csv")
compound_Fuzzy_C_means <- compound_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
compound_Spectral_Clustering <- read_csv(
  "clustering_Spectral_Clustering_0_results.csv")


compound_DBSCAN <- read_csv("clustering_DBSCAN_0_results.csv")
#compound_DBSCAN$Y_0[which(compound_DBSCAN$Y_0==-1)] <- NA
compound_Gaussian_Mixture <- read_csv(
  "clustering_Gaussian_Mixture_0_results.csv")

aggregation_K_means <- read_csv("clustering_K-means_1_results.csv")
aggregation_Fuzzy_C_means <- read_csv("clustering_Fuzzy_C-means_1_results.csv")
aggregation_Fuzzy_C_means <- aggregation_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
aggregation_Spectral_Clustering <- read_csv(
  "clustering_Spectral_Clustering_1_results.csv")
aggregation_DBSCAN <- read_csv("clustering_DBSCAN_1_results.csv")
#aggregation_DBSCAN$Y_0[which(aggregation_DBSCAN$Y_0==-1)] <- NA
aggregation_Gaussian_Mixture <- read_csv(
  "clustering_Gaussian_Mixture_1_results.csv")

pathbased_K_means <- read_csv("clustering_K-means_2_results.csv")
pathbased_Fuzzy_C_means <-
  read_csv("clustering_Fuzzy_C-means_2_results.csv")
pathbased_Fuzzy_C_means <- pathbased_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
pathbased_Spectral_Clustering <- read_csv(
  "clustering_Spectral_Clustering_2_results.csv")
pathbased_DBSCAN <- read_csv("clustering_DBSCAN_2_results.csv")

dbscan_teste <- dbscan(pathbased_DBSCAN[,1:2], 0.35, minPts = 11)
dbscan_teste$cluster
pathbased_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(dbscan_teste$cluster))) +
  geom_point(size = 2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)
pathbased_DBSCAN$Y_0 <- dbscan_teste$cluster
pathbased_DBSCAN$Y_0[which(pathbased_DBSCAN$Y_0== 0)] <- NA

#X_sc <- spectral_clustering(pathbased_DBSCAN[,1:2], nn = 2, n_eig = 2)
#X_sc_kmeans <- kmeans(X_sc, 3)
#pathbased_DBSCAN |> 
#  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
#  ggplot(aes(X0, X1, color = factor(X_sc_kmeans$cluster))) +
#  geom_point(size = 2) +
#  theme_void() +
#  scale_color_discrete(labels = palette, guide = 'none') +
#  xlab(NULL) +
#  ylab(NULL)


#pathbased_DBSCAN$Y_0[which(pathbased_DBSCAN$Y_0==-1)] <- NA
pathbased_Gaussian_Mixture <- read_csv(
  "clustering_Gaussian_Mixture_2_results.csv")

s2_K_means <- read_csv("clustering_K-means_3_results.csv")
s2_Fuzzy_C_means <-
  read_csv("clustering_Fuzzy_C-means_3_results.csv")
s2_Fuzzy_C_means <- s2_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
s2_Spectral_Clustering <- read_csv(
  "clustering_Spectral_Clustering_3_results.csv")
s2_DBSCAN <- read_csv("clustering_DBSCAN_3_results.csv")
#s2_DBSCAN$Y_0[which(s2_DBSCAN$Y_0==-1)] <- NA
s2_Gaussian_Mixture <-
  read_csv("clustering_Gaussian_Mixture_3_results.csv")

flame_K_means <- read_csv("clustering_K-means_4_results.csv")
flame_Fuzzy_C_means <-
  read_csv("clustering_Fuzzy_C-means_4_results.csv")
flame_Fuzzy_C_means <- flame_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
flame_Spectral_Clustering <-
  read_csv("clustering_Spectral_Clustering_4_results.csv")
flame_DBSCAN <- read_csv("clustering_DBSCAN_4_results.csv")
#flame_DBSCAN$Y_0[which(flame_DBSCAN$Y_0==-1)] <- NA
flame_Gaussian_Mixture <-
  read_csv("clustering_Gaussian_Mixture_4_results.csv")


face_DBSCAN <- read_csv("clustering_DBSCAN_5_results.csv")
face_DBSCAN$labels = face_DBSCAN$Y_0

face_K_means <- read_csv("clustering_K-means_5_results.csv")
face_K_means$labels = face_DBSCAN$Y_0
face_Fuzzy_C_means <-
  read_csv("clustering_Fuzzy_C-means_5_results.csv")
face_Fuzzy_C_means <- face_Fuzzy_C_means %>%
  mutate(Cluster = apply(select(., starts_with("Y_")), 1,
                         function(x)
                           which.max(x) - 1))
face_Fuzzy_C_means$labels = face_DBSCAN$Y_0
face_Spectral_Clustering <-
  read_csv("clustering_Spectral_Clustering_5_results.csv")

face_Spectral_Clustering$labels = face_DBSCAN$Y_0
face_Gaussian_Mixture <-
  read_csv("clustering_Gaussian_Mixture_5_results.csv")
face_Gaussian_Mixture$labels = face_DBSCAN$Y_0

palette <-
  c(
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#1a1a1a",
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231"
  )



#################################### compound #########################################


g0 <- compound_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  xlab(NULL) +
  ggtitle("Rótulos") +
  ylab("Compound") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


g1 <- compound_K_means |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  xlab(NULL) +
  ylab(NULL) +
  ggtitle("K-Means") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))



g2 <- compound_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  xlab(NULL) +
  ylab(NULL) +
  ggtitle("Spectral") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


g3 <- compound_DBSCAN |>
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  xlab(NULL) +
  ylab(NULL) +
  ggtitle("DBSCAN") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))




g4 <- compound_Gaussian_Mixture |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  xlab(NULL) +
  ylab(NULL) +
  ggtitle("Gaussian Mixture") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))



g5 <- ggplot(compound_Fuzzy_C_means, aes(X0, X1, color = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  ) +
  ggtitle("Fuzzy C-Means") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


################################### aggregationS ###############################


g0_1 <- aggregation_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab("Aggregation")



g6 <- aggregation_K_means |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g7 <- aggregation_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g8 <- aggregation_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g9 <- aggregation_Gaussian_Mixture |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g10 <- aggregation_Fuzzy_C_means |> 
  ggplot(aes(X0, X1, colour = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  )


############################ pathbased #########################################

g0_2 <- pathbased_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab("PathBased")


g11 <- pathbased_K_means |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g12 <- pathbased_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g13 <- pathbased_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g14 <- pathbased_Gaussian_Mixture |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g15 <- pathbased_Fuzzy_C_means |>  
  ggplot(aes(X0, X1, colour = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  )

#################################### s2 ########################################
g0_3 <- s2_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  #theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab("S2") +
  theme_void()


g16 <- s2_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


s2_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g17 <- s2_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g18 <- s2_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g19 <- s2_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g20 <- s2_Fuzzy_C_means |> 
  ggplot(aes(X0, X1, colour = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  )


#################################### flame #########################################

g0_4 <- flame_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab("Flame")


g21 <- flame_K_means |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g22 <- flame_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g23 <-
  flame_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g24 <- flame_Gaussian_Mixture |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g25 <- flame_Fuzzy_C_means |> 
  ggplot(aes(X0, X1, colour = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  )


#################################### No Structure #########################################

g0_5 <- face_K_means |> 
  ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab("Face")


g26 <- face_K_means |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g27 <- face_Spectral_Clustering |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)


g28 <- face_DBSCAN |> 
  mutate_all( ~ ifelse(. == -1, NA, .)) |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g29 <- face_Gaussian_Mixture |> 
  ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.2) +
  theme_void() +
  scale_color_discrete(labels = palette, guide = 'none') +
  xlab(NULL) +
  ylab(NULL)



g30 <- face_Fuzzy_C_means |> 
  ggplot(aes(X0, X1, colour = factor(Cluster))) +
  geom_point(size = 0.2) +
  theme_void() +
  xlab(NULL) +
  ylab(NULL) +
  scale_color_discrete(
    labels = palette,
    guide = 'none'
  )




plots <-
  list(
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g0_1,
    g6,
    g7,
    g8,
    g9,
    g10,
    g0_2,
    g11,
    g12,
    g13,
    g14,
    g15,
    g0_3,
    g16,
    g17,
    g18,
    g19,
    g20,
    g0_4,
    g21,
    g22,
    g23,
    g24,
    g25,
    g0_5,
    g26,
    g27,
    g28,
    g29,
    g30
  )

for (i in 1:length(plots)) {
  plots[[i]] <- plots[[i]] + theme(
    legend.position = 'none',
    panel.border =
      element_rect(
        color = "black",
        linewidth = 0.2,
        fill = NA
      )
  )
}

g <- plot_grid(plotlist = plots)

ggsave("clustering_plot.pdf", g, width = 6, height  = 6)

