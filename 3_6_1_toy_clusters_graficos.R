library(readr)
library(ggplot2)
library("MixSim")
library(latex2exp)
library("fclust")
library("plotly")
library(cowplot)
library(dplyr)
options(scipen=999)
setwd("F:/TG2/results/toy_results")


circle_K_means <- read_csv("toydataset_K-means_0_results.csv")
circle_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_0_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
circle_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_0_results.csv")
circle_DBSCAN <- read_csv("toydataset_DBSCAN_0_results.csv")
circle_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_0_results.csv")
moon_K_means <- read_csv("toydataset_K-means_1_results.csv")
moon_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_1_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
moon_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_1_results.csv")
moon_DBSCAN <- read_csv("toydataset_DBSCAN_1_results.csv")
moon_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_1_results.csv")
varied_K_means <- read_csv("toydataset_K-means_2_results.csv")
varied_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_2_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
varied_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_2_results.csv")
varied_DBSCAN <- read_csv("toydataset_DBSCAN_2_results.csv")
varied_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_2_results.csv")
aniso_K_means <- read_csv("toydataset_K-means_3_results.csv")
aniso_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_3_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
aniso_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_3_results.csv")
aniso_DBSCAN <- read_csv("toydataset_DBSCAN_3_results.csv")
aniso_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_3_results.csv")
blobs_K_means <- read_csv("toydataset_K-means_4_results.csv")
blobs_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_4_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
blobs_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_4_results.csv")
blobs_DBSCAN <- read_csv("toydataset_DBSCAN_4_results.csv")
blobs_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_4_results.csv")
no_struct_K_means <- read_csv("toydataset_K-means_5_results.csv")
no_struct_Fuzzy_C_means <- read_csv("toydataset_Fuzzy_C-means_5_results.csv")%>%
  mutate(
    Y = apply(select(., starts_with("Y_")), 1,
              function(x) which.max(x) - 1)
  )
no_struct_Spectral_Clustering <- read_csv("toydataset_Spectral_Clustering_5_results.csv")
no_struct_DBSCAN <- read_csv("toydataset_DBSCAN_5_results.csv")
no_struct_Gaussian_Mixture <- read_csv("toydataset_Gaussian_Mixture_5_results.csv")





#################################### CIRCLE #########################################


g0 <- circle_K_means |> ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab("noise_circles") +
  ggtitle("Rótulos") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))

circle_K_means_RI = round(RandIndex(circle_K_means$labels, circle_K_means$Y_0)$AR, 2)
circle_K_means_CP = round(ClassProp(circle_K_means$labels, circle_K_means$Y_0), 2)
g1 <- circle_K_means |> ggplot(aes(X0, X1, color = as.factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) +
  ggtitle("K-Means") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


circle_Spectral_Clustering_RI = round(RandIndex(circle_Spectral_Clustering$labels, circle_Spectral_Clustering$Y_0)$AR, 2)
circle_Spectral_Clustering_CP = round(ClassProp(circle_Spectral_Clustering$labels, circle_Spectral_Clustering$Y_0), 2)
g2 <- circle_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) +
  ggtitle("Spectral") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))

circle_DBSCAN_RI = round(RandIndex(circle_DBSCAN$labels, circle_DBSCAN$Y_0)$AR, 2)
circle_DBSCAN_CP = round(ClassProp(circle_DBSCAN$labels, circle_DBSCAN$Y_0), 2)
g3 <- circle_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) +
  ggtitle("DBSCAN") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


circle_Gaussian_Mixture_RI = round(RandIndex(circle_Gaussian_Mixture$labels, circle_Gaussian_Mixture$Y_0)$AR, 2)
circle_Gaussian_Mixture_CP = round(ClassProp(circle_Gaussian_Mixture$labels, circle_Gaussian_Mixture$Y_0), 2)
g4 <- circle_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) +
  ggtitle("Gaussian Mixture") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


circle_Fuzzy_C_means$clusters = apply(cbind(circle_Fuzzy_C_means$Y_0,
                                          circle_Fuzzy_C_means$Y_1,
                                          circle_Fuzzy_C_means$Y_2),
                                    1,
                                    which.max) - 1
circle_Fuzzy_C_means_RI = round(RI.F(circle_Fuzzy_C_means$labels, cbind(circle_Fuzzy_C_means[,c("Y_0","Y_1","Y_2")]), "minimum"), 2)
circle_Fuzzy_C_means_CP = round(ClassProp(circle_Fuzzy_C_means$labels, circle_Fuzzy_C_means$clusters), 2)
n <- nrow(circle_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- circle_Fuzzy_C_means$Y_0[i]
  g <- circle_Fuzzy_C_means$Y_1[i]
  b <- circle_Fuzzy_C_means$Y_2[i]
  colors[i] <- rgb(r, g, b)
}
circle_Fuzzy_C_means$colors <- colors
g5 <- ggplot(circle_Fuzzy_C_means, aes(X0, X1, color = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL)+
  scale_color_identity(guide = 'none') +
  ggtitle("Fuzzy C-Means") +
  theme(plot.title = element_text(hjust = 0.5, size = 8))


#################################################################### MOONS ###############################


g0_1 <- moon_K_means |> ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  scale_color_discrete(labels = c("red", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab("noisy_moons") 

moon_K_means_RI = round(RandIndex(moon_K_means$labels, moon_K_means$Y_0)$AR, 2)
moon_K_means_CP = round(ClassProp(moon_K_means$labels, moon_K_means$Y_0), 2)
g6 <- moon_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


moon_Spectral_Clustering_RI = round(RandIndex(moon_Spectral_Clustering$labels, moon_Spectral_Clustering$Y_0)$AR, 2)
moon_Spectral_Clustering_CP = round(ClassProp(moon_Spectral_Clustering$labels, moon_Spectral_Clustering$Y_0), 2)
g7 <- moon_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 

moon_DBSCAN_RI = round(RandIndex(moon_DBSCAN$labels, moon_DBSCAN$Y_0)$AR, 2)
moon_DBSCAN_CP = round(ClassProp(moon_DBSCAN$labels, moon_DBSCAN$Y_0), 2)
g8 <- moon_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


moon_Gaussian_Mixture_RI = round(RandIndex(moon_Gaussian_Mixture$labels, moon_Gaussian_Mixture$Y_0)$AR, 2)
moon_Gaussian_Mixture_CP = round(ClassProp(moon_Gaussian_Mixture$labels, moon_Gaussian_Mixture$Y_0), 2)
g9 <- moon_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


moon_Fuzzy_C_means$clusters = apply(cbind(moon_Fuzzy_C_means$Y_0,
                                            moon_Fuzzy_C_means$Y_1),
                                      1,
                                      which.max) - 1
moon_Fuzzy_C_means_RI = round(RI.F(moon_Fuzzy_C_means$labels, cbind(moon_Fuzzy_C_means[,c("Y_0","Y_1")]), "minimum"), 2)
moon_Fuzzy_C_means_CP = round(ClassProp(moon_Fuzzy_C_means$labels, moon_Fuzzy_C_means$clusters), 2)
n <- nrow(moon_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- moon_Fuzzy_C_means$Y_0[i]
  g <- moon_Fuzzy_C_means$Y_1[i]
  colors[i] <- rgb(r, g, 0)
}

moon_Fuzzy_C_means$colors <- colors

# plot the points with color
g10 <- moon_Fuzzy_C_means |> ggplot(aes(X0, X1, colour = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL) +
  scale_color_identity(guide = 'none') 


#################################### VARIED #########################################

g0_2 <- varied_K_means |> ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab("varied") 

varied_K_means_RI = round(RandIndex(varied_K_means$labels, varied_K_means$Y_0)$AR, 2)
varied_K_means_CP = round(ClassProp(varied_K_means$labels, varied_K_means$Y_0), 2)
g11 <- varied_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


varied_Spectral_Clustering_RI = round(RandIndex(varied_Spectral_Clustering$labels, varied_Spectral_Clustering$Y_0)$AR, 2)
varied_Spectral_Clustering_CP = round(ClassProp(varied_Spectral_Clustering$labels, varied_Spectral_Clustering$Y_0), 2)
g12 <- varied_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 

varied_DBSCAN_RI = round(RandIndex(varied_DBSCAN$labels, varied_DBSCAN$Y_0)$AR, 2)
varied_DBSCAN_CP = round(ClassProp(varied_DBSCAN$labels, varied_DBSCAN$Y_0), 2)
g13 <- varied_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 



varied_Gaussian_Mixture_RI = round(RandIndex(varied_Gaussian_Mixture$labels, varied_Gaussian_Mixture$Y_0)$AR, 2)
varied_Gaussian_Mixture_CP = round(ClassProp(varied_Gaussian_Mixture$labels, varied_Gaussian_Mixture$Y_0), 2)
g14 <- varied_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


varied_Fuzzy_C_means$clusters = apply(cbind(varied_Fuzzy_C_means$Y_0,
                                            varied_Fuzzy_C_means$Y_1,
                                            varied_Fuzzy_C_means$Y_2),
                                      1,
                                      which.max) - 1
varied_Fuzzy_C_means_RI = round(RI.F(varied_Fuzzy_C_means$labels, cbind(varied_Fuzzy_C_means[,c("Y_0","Y_1","Y_2")]), "minimum"), 2)
varied_Fuzzy_C_means_CP = round(ClassProp(varied_Fuzzy_C_means$labels, varied_Fuzzy_C_means$clusters), 2)
n <- nrow(varied_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- varied_Fuzzy_C_means$Y_0[i]
  g <- varied_Fuzzy_C_means$Y_1[i]
  b <- varied_Fuzzy_C_means$Y_2[i]
  colors[i] <- rgb(r, g, b)
}
varied_Fuzzy_C_means$colors <- colors
g15 <- varied_Fuzzy_C_means |> ggplot(aes(X0, X1, color = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL)+
  scale_color_identity(guide = 'none')

#################################### ANISO #########################################
g0_3 <- aniso_K_means |>ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  scale_color_discrete(labels = c("red", "blue", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab("aniso") 


aniso_K_means_RI = round(RandIndex(aniso_K_means$labels, aniso_K_means$Y_0)$AR, 2)
aniso_K_means_CP = round(ClassProp(aniso_K_means$labels, aniso_K_means$Y_0), 2)
g16 <- aniso_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "blue", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


aniso_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 2) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "blue", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


aniso_Spectral_Clustering_RI = round(RandIndex(aniso_Spectral_Clustering$labels, aniso_Spectral_Clustering$Y_0)$AR, 2)
aniso_Spectral_Clustering_CP = round(ClassProp(aniso_Spectral_Clustering$labels, aniso_Spectral_Clustering$Y_0), 2)
g17 <- aniso_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "blue", "green"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 

aniso_DBSCAN_RI = round(RandIndex(aniso_DBSCAN$labels, aniso_DBSCAN$Y_0)$AR, 2)
aniso_DBSCAN_CP = round(ClassProp(aniso_DBSCAN$labels, aniso_DBSCAN$Y_0), 2)
g18 <- aniso_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


aniso_Gaussian_Mixture_RI = round(RandIndex(aniso_Gaussian_Mixture$labels, aniso_Gaussian_Mixture$Y_0)$AR, 2)
aniso_Gaussian_Mixture_CP = round(ClassProp(aniso_Gaussian_Mixture$labels, aniso_Gaussian_Mixture$Y_0), 2)
g19 <- aniso_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


aniso_Fuzzy_C_means$clusters = apply(cbind(aniso_Fuzzy_C_means$Y_0,
                                            aniso_Fuzzy_C_means$Y_1,
                                            aniso_Fuzzy_C_means$Y_2),
                                      1,
                                      which.max) - 1
aniso_Fuzzy_C_means_RI = round(RI.F(aniso_Fuzzy_C_means$labels, cbind(aniso_Fuzzy_C_means[,c("Y_0","Y_1","Y_2")]), "minimum"), 2)
aniso_Fuzzy_C_means_CP = round(ClassProp(aniso_Fuzzy_C_means$labels, aniso_Fuzzy_C_means$clusters), 2)
n <- nrow(aniso_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- aniso_Fuzzy_C_means$Y_0[i]
  g <- aniso_Fuzzy_C_means$Y_1[i]
  b <- aniso_Fuzzy_C_means$Y_2[i]
  colors[i] <- rgb(r, g, b)
}
aniso_Fuzzy_C_means$colors <- colors
g20 <- aniso_Fuzzy_C_means |> ggplot(aes(X0, X1, color = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL)+
  scale_color_identity(guide = 'none') 


#################################### Blobs #########################################

g0_4 <- blobs_K_means |> ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab("blobs") 

blobs_K_means_RI = round(RandIndex(blobs_K_means$labels, blobs_K_means$Y_0)$AR, 2)
blobs_K_means_CP = round(ClassProp(blobs_K_means$labels, blobs_K_means$Y_0), 2)
g21 <- blobs_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


blobs_Spectral_Clustering_RI = round(RandIndex(blobs_Spectral_Clustering$labels, blobs_Spectral_Clustering$Y_0)$AR, 2)
blobs_Spectral_Clustering_CP = round(ClassProp(blobs_Spectral_Clustering$labels, blobs_Spectral_Clustering$Y_0), 2)
g22 <- blobs_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 

blobs_DBSCAN_RI = round(RandIndex(blobs_DBSCAN$labels, blobs_DBSCAN$Y_0)$AR, 2)
blobs_DBSCAN_CP = round(ClassProp(blobs_DBSCAN$labels, blobs_DBSCAN$Y_0), 2)
g23 <- blobs_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


blobs_Gaussian_Mixture_RI = round(RandIndex(blobs_Gaussian_Mixture$labels, blobs_Gaussian_Mixture$Y_0)$AR, 2)
blobs_Gaussian_Mixture_CP = round(ClassProp(blobs_Gaussian_Mixture$labels, blobs_Gaussian_Mixture$Y_0), 2)
g24 <- blobs_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL)


blobs_Fuzzy_C_means$clusters = apply(cbind(blobs_Fuzzy_C_means$Y_0,
                                           blobs_Fuzzy_C_means$Y_1,
                                           blobs_Fuzzy_C_means$Y_2),
                                     1,
                                     which.max) - 1
blobs_Fuzzy_C_means_RI = round(RI.F(blobs_Fuzzy_C_means$labels, cbind(blobs_Fuzzy_C_means[,c("Y_0","Y_1","Y_2")]), "minimum"), 2)
blobs_Fuzzy_C_means_CP = round(ClassProp(blobs_Fuzzy_C_means$labels, blobs_Fuzzy_C_means$clusters), 2)
n <- nrow(blobs_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- blobs_Fuzzy_C_means$Y_0[i]
  g <- blobs_Fuzzy_C_means$Y_1[i]
  b <- blobs_Fuzzy_C_means$Y_2[i]
  colors[i] <- rgb(r, g, b)
}
blobs_Fuzzy_C_means$colors <- colors
g25 <- blobs_Fuzzy_C_means |> ggplot(aes(X0, X1, color = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL) +
  scale_color_identity(guide = 'none') 


#################################### No Structure #########################################

g0_5 <- no_struct_K_means |> ggplot(aes(X0, X1, color = factor(labels))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab("no_struct") 

no_struct_K_means_RI = round(RandIndex(no_struct_K_means$labels, no_struct_K_means$Y_0)$AR, 2)
no_struct_K_means_CP = round(ClassProp(no_struct_K_means$labels, no_struct_K_means$Y_0), 2)
g26 <- no_struct_K_means |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


no_struct_Spectral_Clustering_RI = round(RandIndex(no_struct_Spectral_Clustering$labels, no_struct_Spectral_Clustering$Y_0)$AR, 2)
no_struct_Spectral_Clustering_CP = round(ClassProp(no_struct_Spectral_Clustering$labels, no_struct_Spectral_Clustering$Y_0)/1000, 2)
g27 <- no_struct_Spectral_Clustering |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 

no_struct_DBSCAN_RI = round(RandIndex(no_struct_DBSCAN$labels, no_struct_DBSCAN$Y_0)$AR, 2)
no_struct_DBSCAN_CP = round(ClassProp(no_struct_DBSCAN$labels, no_struct_DBSCAN$Y_0)/1000, 2)
g28 <- no_struct_DBSCAN |> mutate_all(~ifelse(. == -1, NA, .)) |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


no_struct_Gaussian_Mixture_RI = round(RandIndex(no_struct_Gaussian_Mixture$labels, no_struct_Gaussian_Mixture$Y_0)$AR, 2)
no_struct_Gaussian_Mixture_CP = round(ClassProp(no_struct_Gaussian_Mixture$labels, no_struct_Gaussian_Mixture$Y_0)/1000, 2)
g29 <- no_struct_Gaussian_Mixture |> ggplot(aes(X0, X1, color = factor(Y_0))) +
  geom_point(size = 0.5) + 
  theme_void() + 
  scale_color_discrete(labels = c("red", "green", "blue"),guide = 'none') +
  xlab(NULL) + 
  ylab(NULL) 


no_struct_Fuzzy_C_means$clusters = apply(cbind(no_struct_Fuzzy_C_means$Y_0,
                                           no_struct_Fuzzy_C_means$Y_1,
                                           no_struct_Fuzzy_C_means$Y_2),
                                     1,
                                     which.max) - 1
no_struct_Fuzzy_C_means_RI = round(RI.F(no_struct_Fuzzy_C_means$labels, cbind(no_struct_Fuzzy_C_means[,c("Y_0","Y_1","Y_2")]), "minimum"), 2)
no_struct_Fuzzy_C_means_CP = round(ClassProp(no_struct_Fuzzy_C_means$labels, no_struct_Fuzzy_C_means$clusters)/1000, 2)
n <- nrow(no_struct_Fuzzy_C_means)
colors <- numeric(n)
for (i in 1:n) {
  r <- no_struct_Fuzzy_C_means$Y_0[i]
  g <- no_struct_Fuzzy_C_means$Y_1[i]
  b <- no_struct_Fuzzy_C_means$Y_2[i]
  colors[i] <- rgb(r, g, b)
}
no_struct_Fuzzy_C_means$colors <- colors
g30 <- no_struct_Fuzzy_C_means |> ggplot(aes(X0, X1, color = colors)) +
  geom_point(size = 0.5) + 
  theme_void() + 
  xlab(NULL) + 
  ylab(NULL)+
  scale_color_identity(guide = 'none') 




plots <- list(g0, g1, g2, g3, g4, g5, g0_1, g6, g7, g8, g9, g10, g0_2, g11, g12, 
              g13, g14, g15, g0_3, g16, g17, g18, g19, g20, g0_4, g21, g22, g23,
              g24, g25,g0_5, g26, g27, g28, g29, g30)

for (i in 1:length(plots)) {
  plots[[i]] <- plots[[i]] + theme(legend.position = 'none', 
                                   panel.border = 
                                     element_rect(color = "black", 
                                                  linewidth = 0.2, 
                                                  fill = NA
                                                  )
                                   )
}

g <- plot_grid(plotlist = plots)

ggsave("plot2.png", g, width = 6, height  = 6)


