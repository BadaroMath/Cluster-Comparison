
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





gridn <- expand.grid(BarOmega = 0,
                       K = 3,
                       P = 5,
                       N = c(seq(1000, 1000000, 1000)))

apply(gridn, 1, function(x) operator(x[1], x[2], x[3], x[4]))


################ General (individual)  ##############


for (BarOmega in seq(0, 0.61, 0.01)){
  operator(BarOmega, 3, 5, 5000)
}

for (K in seq(2, 40, 1)){
  operator(0, K, 3, 5000)
}

for (n in c(seq(115000, 1000000, 1000))){
  operator(0, 3, 5, n)
}

for (p in c(seq(2,80, 2), seq(30, 200, 10))){
  operator(0, 3, p, 5000)
}




################ General (individual) BarOmega = 0.05  ##############

for (K in seq(2, 40, 1)){
  operator(0.05, K, 3, 5000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.05, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 70, 10))){
  operator(0.05, 3, p, 5000)
}

################ General (individual) BarOmega = 0.1  ##############

for (K in seq(2, 40, 1)){
  operator(0.1, K, 3, 5000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.1, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 50, 10))){
  operator(0.1, 3, p, 5000)
}



################ General (individual) BarOmega = 0.15  ##############
for (K in seq(2, 40, 1)){
  operator(0.15, K, 3, 5000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.15, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 40, 10))){
  operator(0.15, 3, p, 5000)
}



################ General (individual) BarOmega = 0.20  ##############

for (K in seq(2, 40, 1)){
  operator(0.2, K, 3, 5000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.2, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 30, 10))){
  operator(0.2, 3, p, 5000)
}



################ Cluster & Components  ##############


for (K in c(3, 5, 7, 10, 15, 20, 25)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    operator(0, K, p, 10000)
  }
}



################ Obs & Components  ##############


for (n in c(1000, 5000, 50000, 1000000)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    operator(0, 3, p, n)
  }
}


