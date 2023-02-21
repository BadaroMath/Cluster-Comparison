
library("MixSim")
setwd("C:/Users/maatt/Documents/Estatística/TG2")
options(scipen=999)
###################### Simulation ##########################################
###############################################################################


operator <- function(BarOmega, K, p, n){
  Q <- MixSim(BarOmega = BarOmega, K = K, p = p, resN = 100000)
  A <- simdataset(n = n, Pi = Q$Pi, Mu = Q$Mu, S = Q$S)
  write.csv(A, paste("J:/simulações/simdataset_",
                     BarOmega,"_", K,"_", p,"_", n,".csv", sep=""), row.names=FALSE)
  cat("BarOmega=",BarOmega,"Clusters=",K, "Components=",p, "Obs=",n, "\n")
}

for (n in c(seq(100, 5000, 500),seq(5000, 100000, 10000), seq(100000, 500000, 100000))) {
  for (p in c(3,5,7,10,15,seq(20,100, 20))) {
    for (BarOmega in seq(0, 0.50, .05)) {
      for (K in c(seq(2,10,1), seq(15, 30, 5))) {
        operator(BarOmega, K, p, n)
      }
    }
  }
}



################ General (individual)  ##############


for (BarOmega in seq(0, 0.6, 0.01)){
  operator(BarOmega, 3, 5, 2000)
}

for (K in seq(2, 40, 1)){
  operator(0, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 200, 10))){
  operator(0, 3, p, 2000)
}




################ General (individual) BarOmega = 0.05  ##############

for (K in seq(2, 40, 1)){
  operator(0.05, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.05, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 70, 10))){
  operator(0.05, 3, p, 2000)
}

################ General (individual) BarOmega = 0.1  ##############

for (K in seq(2, 40, 1)){
  operator(0.1, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.1, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 70, 10))){
  operator(0.1, 3, p, 2000)
}


################ General (individual) BarOmega = 0.15  ##############
for (K in seq(2, 40, 1)){
  operator(0.15, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.15, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 40, 10))){
  operator(0.15, 3, p, 2000)
}



################ General (individual) BarOmega = 0.20  ##############

for (K in seq(2, 40, 1)){
  operator(0.2, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  operator(0.2, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 40, 10))){
  operator(0.2, 3, p, 2000)
}



################ Cluster & Components  ##############


for (K in c(3, 5, 7, 10)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    operator(0, K, p, 1000)
  }
}



################ Obs & Components  ##############


for (n in seq(1000, 10000, 1000)){
  for (p in c(3, 5, 8, 10, 15, 30, 50)){
    operator(0, K, p, 1000)
  }
}


