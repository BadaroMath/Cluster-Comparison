library(devtools)
install_github("elbamos/clusteringdatasets")


data("Compound")
data("Aggregation")
data("pathbased")
data("s2")
data("flame")
data("face")


write_csv(Compound, "F:/TG2/results/clusteringdatasets/Compound.csv")
write_csv(Aggregation, "F:/TG2/results/clusteringdatasets/Aggregation.csv")
write_csv(pathbased, "F:/TG2/results/clusteringdatasets/pathbased.csv")
write_csv(s2, "F:/TG2/results/clusteringdatasets/s2.csv")
write_csv(flame, "F:/TG2/results/clusteringdatasets/flame.csv")
write_csv(face, "F:/TG2/results/clusteringdatasets/face.csv")
