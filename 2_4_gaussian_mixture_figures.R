library(mclust)
library(ggplot2)
library(dplyr)


set.seed(12345)


############## Figura 2.6


n <- 200
R1 <- matrix(c(1, 1,
              1, 69), nrow = 2, ncol = 2)

mu1 <- c(X = 10, Y = 30)
mu2 <- c(X = 6, Y = 9)

R2 <- matrix(c(0.1, 1,
              1, 36), nrow = 2, ncol = 2)
pt1 = MASS::mvrnorm(n, mu = mu1, Sigma = R1)
pt2 = MASS::mvrnorm(n, mu = mu2, Sigma = R2)
simul = as.data.frame(rbind(pt1, pt2))


simul_uni = simul[,2]
n <- length (simul_uni)
simul_uni.Mclust <- Mclust(simul_uni,model="V",G=2)
# Plot densities:
x <- seq (from=min(simul_uni),to=max(simul_uni),length=10000)
den1 <- dnorm (x,mean=simul_uni.Mclust$parameters$mean[1],
               sd=sqrt(simul_uni.Mclust$parameters$variance$sigmasq
                       [1]))
den2 <- dnorm (x,mean=simul_uni.Mclust$parameters$mean[2], 
               sd=sqrt(simul_uni.Mclust$parameters$variance$sigmasq[2]))
tau1 <- simul_uni.Mclust$parameters$pro [1]
tau2 <- simul_uni.Mclust$parameters$pro [2]
dens <- c(tau1*den1 + tau2*den2)

density = data.frame(x, dens, tau1*den1, tau2*den2)
colnames(density) = c("x", "dens", "comp1", "comp2")

# Simulate and plot a sample from the distribution:

n <- 1000
sim.results <- simV(simul_uni.Mclust$parameters, n, seed = 0)
group = simV(simul_uni.Mclust$parameters, n, seed = 0) %>% 
  as.data.frame() 
colnames(group) = c("Comp", "x")
for (i in 1:length(group$Comp)) {
  group$Comp[i] = ifelse(group$Comp[i] == 1, "Componente 1", "Componente 2")
}

group$y = -max(dens)/20
group$Comp = as.factor(group$Comp)

ggplot(data = density) + 
  geom_line(aes(x= x, y = dens, color = "Mistura"), 
            size = 1, color = "black", linetype = 1) +
  geom_line(aes(x=x, y = comp1), 
            color = "red",linetype = "dashed") +
  geom_line(aes(x=x, y = comp2), 
            color = "blue",linetype = "dashed") + 
  geom_point(data = group, aes(x = x, y = y, 
                               color = Comp), shape = 19) +
  scale_color_manual(values = c("Mistura Gaussiana" = "black", 
                                "Componente 1" = "red", 
                                "Componente 2" = "blue"))+
  labs(title="Densidade do modelo de mistura gaussiana 
       unidimensional para dois componentes",
       x ="y", y = "Densidade de Probabilidade", color = "Componente") + 
  theme_bw() +
  theme(
    plot.title = element_text(size=14, face="bold.italic"),
    axis.title.x = element_text(size=12, face="bold"),
    axis.title.y = element_text( size=12, face="bold")
  )


############## Figura 2.7


n <- length (simul [,1])
simul.Mclust2den <- densityMclust (simul , model="VVV",G=2)
# Simulate a sample from the density
sim.results <- simVVV (simul.Mclust2den$parameters ,n,seed =0)
ysim <- sim.results[,c(2,3)]
groupsim <- sim.results[,"group"]
ysim1 <- ysim[groupsim ==1,]
ysim2 <- ysim[groupsim ==2,]
# Plot the density and simulated points

plot(simul.Mclust2den , col="black",
      xlab="Dimension 1",
      ylab="Dimension 2", 
      xlim=c(min(ysim [,1])-0.5,max(ysim [,1])), 
      ylim = c(min(ysim[,2])-2,max(ysim [,2])), 
      levels = round (quantile(
        simul.Mclust2den$density, 
        probs = c(0.05, 0.25, 0.5, 0.75, 0.95 , 0.99)) 
        ,3))
points (ysim1 ,col="red",pch =19)
points (ysim2 ,col="blue",pch =19)
legend (x=4.4 ,y=55 , 
        legend=c("Contornos da densidade","Componente 1","Componente 2"),
        col=c("black","red","blue"),lty=c(1,0,0), 
        lwd=c(2,0,0), pch=c(NA ,19 ,19) )