





metrics <- function(Method, BarOmega, K, P, N){
  cat("--------------------------", 
      "\n", "Method: ", Method, "\n", "BarOmega: ", 
      BarOmega, "\n", "K: ", K, "\n", "P: ", P, "\n", "N: ", N)
  labeled <- read_csv(paste("F:/TG2/results/", Method, "_simuls/", Method, "_", 
                            BarOmega,"_", K,"_", P,"_", N, "_results",".csv", 
                            sep=""), show_col_types = FALSE)
  
  results <- data.frame(
                        Method = Method,
                        BarOmega = BarOmega,
                        K = K,
                        P = P,
                        N = N,
                        RI = round(RandIndex(labeled$id, labeled$Y)$R, 3),
                        AR = round(RandIndex(labeled$id, labeled$Y)$AR, 3),
                        ClassProp = round(ClassProp(labeled$id, labeled$Y), 3),
                        VI = round(VarInf(labeled$id, labeled$Y), 3)
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
                round(VarInf(labeled$id, labeled$Y), 3))
  ),
  file=paste("F:/TG2/results/", Method, "_simuls/", Method, "_metrics",".txt", 
             sep=""), append = TRUE, sep="\n")
}




for (BarOmega in seq(0, 0.6, 0.01)){
  metrics("kmeans", BarOmega, 3, 5, 2000)
}

for (K in seq(2, 40, 1)){
  metrics("kmeans", 0, K, 3, 2000)
}

for (n in c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))){
  metrics("kmeans", 0, 3, 5, n)
}

for (p in c(seq(2,20, 2), seq(30, 200, 10))){
  metrics("kmeans", 0, 3, p, 2000)
}




kmeans <- read_csv("F:/TG2/results/kmeans_simuls/kmeans_metrics.txt",
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
                     "VI"
                     )
                   )


kmeans |> ggplot() +
  geom_point(aes(BarOmega, AR), color = "red") +
  geom_point(aes(BarOmega, RI), color = "green") + 
  geom_point(aes(BarOmega, ClassProp), color = "blue")


kmeans |> ggplot() +
  geom_point(aes(K, AR), color = "red") +
  geom_point(aes(K, RI), color = "green") + 
  geom_point(aes(K, ClassProp), color = "blue")


kmeans |> ggplot() +
  geom_point(aes(N, AR), color = "red") +
  geom_point(aes(N, RI), color = "green") + 
  geom_point(aes(N, ClassProp), color = "blue")


