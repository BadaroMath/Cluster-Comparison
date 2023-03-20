
library("MixSim")
setwd("F:/TG2/results/simulações")
options(scipen=999)
###################### Simulation ##########################################
###############################################################################


operator <- function(BarOmega, K, p, n){
  Q <- MixSim(BarOmega = BarOmega, K = K, p = p, resN = 100000)
  A <- simdataset(n = n, Pi = Q$Pi, Mu = Q$Mu, S = Q$S)
  write.csv(A, paste("simdataset_",
                     BarOmega,"_", K,"_", p,"_", n,".csv", sep=""), row.names=FALSE)
  cat("BarOmega=",BarOmega,"Clusters=",K, "Components=",p, "Obs=",n, "\n")
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


apply(grid_bomg, 1, function(x) operator(x[1], x[2], x[3], x[4]))

