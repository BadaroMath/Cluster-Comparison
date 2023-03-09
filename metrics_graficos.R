

library(latex2exp)
library(ggplot2)
library(dplyr)

setwd("F:/TG2/results/Graficos")
metric <- read_csv("F:/TG2/results/resultados_metricas/total_metrics.txt",
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
                      "DI",
                      "SI"
                      )
                    ) %>% 
  mutate(Method = case_when(
    Method == "spectral" ~ "Spectral",
    Method == "K_means" ~ "K-Means",
    Method == "Gaussian_Mixture" ~ "Gaussian Mixture",
    Method == "Fuzzy_C_means" ~ "FCM",
    TRUE ~ Method
  )) %>%
  group_by(Method, BarOmega, K, P, N) %>%
  summarise(AR = max(AR),
            RI = max(RI), 
            AR = max(AR), 
            ClassProp = max(ClassProp),
            VI = min(VI),
            DI = max(DI),
            SI = max(SI))


times <- read_csv("F:/TG2/results/resultados_clusters/time_results.txt",
                  show_col_types = FALSE, 
                  col_names = c(
                    "Method", 
                    "BarOmega", 
                    "K", 
                    "P", 
                    "N", 
                    "time"
                  )
                ) %>% 
  mutate(Method = case_when(
    Method == "K-means" ~ "K-Means",
    Method == "Gaussian_Mixture" ~ "Gaussian Mixture",
    Method == "Fuzzy_C-means" ~ "FCM",
    TRUE ~ Method
  )) %>%
  group_by(Method, BarOmega, K, P, N) %>%
  summarise(time = min(time))

metrics <- tibble(merge(metric, times, by = c("Method", "BarOmega", "K", "P", "N"), all.x = TRUE))





################ BarOmega Variation  ##############



  
ARomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000) %>% 
  mutate(Peso = ifelse(AR > 0.9, 100, 1)) %>%
  ggplot(aes(x = BarOmega, y = AR, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  geom_smooth(aes(weight=Peso), span = 0.4, se = FALSE) +
  #stat_smooth(aes(weight=Peso),method = "lm",
  #            formula = y ~ poly(x, 3), se = FALSE) +
  ylab("Índice de Rand Ajustado")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$.")) 
  
ggsave("ARomega.pdf", 
       plot = ARomega,
       width = 6,
       height = 3
       )


CPomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000) %>% 
  ggplot(aes(x = BarOmega, y = ClassProp, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Proporção de classificações corretas.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$.")) +
  ylim(c(0,1))

ggsave("CPomega.pdf", 
       plot = CPomega,
       width = 6,
       height = 3
)

DIomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000, DI < 1) %>% 
  ggplot(aes(x = BarOmega, y = DI, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Dunn.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("DIomega.pdf", 
       plot = DIomega,
       width = 6,
       height = 3
)


SIomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000) %>% 
  ggplot(aes(x = BarOmega, y = SI, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Coeficiente de Silhueta.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("SIomega.pdf", 
       plot = SIomega,
       width = 6,
       height = 3
)

VIomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000, VI < 4) %>%
  mutate(Peso = ifelse(VI <= 0.1, 100, 1)) %>%
  mutate(Peso = ifelse(VI > 2.8, 10, Peso)) %>%
  ggplot(aes(x = BarOmega, y = VI, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  #geom_smooth(aes(weight=Peso), span = 1, se = FALSE) +
  stat_smooth(aes(weight=Peso),method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("VIomega.pdf", 
       plot = VIomega,
       width = 6,
       height = 3
)

TIMEomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000
         ) %>% 
  ggplot(aes(x = BarOmega, y = time, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Tempo de execução em segundos.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("TIMEomega.pdf", 
       plot = TIMEomega,
       width = 6,
       height = 3
)



################ Clusters Variation  ##############

ARcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  ggplot(aes(x = K, y = AR, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de agrupamentos.") +
  ylim(c(0,1))

ggsave("ARcluster.pdf", 
       plot = ARcluster,
       width = 6,
       height = 3
)

RIcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  ggplot(aes(x = K, y = RI, color = Method)) +
  geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3), se = FALSE) +
  ylab("Índice de Rand.")+
  xlab("N° de agrupamentos.") 

ggsave("RIcluster.pdf", 
       plot = RIcluster,
       width = 6,
       height = 3
)

VIcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  mutate(VI = case_when(VI == Inf ~ 1, TRUE ~ VI)) %>%
  ggplot(aes(x = K, y = VI, color = Method)) +
  geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab("N° de agrupamentos.") +
  ylim(c(0,1))

ggsave("VIcluster.pdf", 
       plot = VIcluster,
       width = 6,
       height = 3
)


DIcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  ggplot(aes(x = K, y = DI, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Dunn.")+
  xlab("N° de agrupamentos.") +
  ylim(c(0,1))

ggsave("DIcluster.pdf", 
       plot = DIcluster,
       width = 6,
       height = 3
)


SIcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  ggplot(aes(x = K, y = SI, color = Method)) +
  geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 3), se = FALSE) +
  ylab("Coeficiente de Silhueta.")+
  xlab("N° de agrupamentos.") +
  ylim(c(0,1))

ggsave("SIcluster.pdf", 
       plot = SIcluster,
       width = 6,
       height = 3
)

TIMEcluster <- metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>% 
  ggplot(aes(x = K, y = time, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Tempo de execução em segundos.")+
  xlab("N° de agrupamentos.")

ggsave("TIMEcluster.pdf", 
       plot = TIMEcluster,
       width = 6,
       height = 3
)


################ Dimensions Variation  ##############


ARdim <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    N == 5000,
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
    !(P %in% c(60,70,80,160))
  ) %>% 
  ggplot(aes(x = P, y = AR, color = Method)) +
  #geom_point(size=1) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de dimensões.") +
  ylim(c(0.5, 1))

ggsave("ARdim.pdf", 
       plot = ARdim,
       width = 6,
       height = 3
)


RIdim <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    N == 5000,
    RI > 0.7,
    !(P %in% c(60,70,80,160))
  ) %>% 
  ggplot(aes(x = P, y = RI, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Rand.")+
  xlab("N° de dimensões.") +
  ylim(c(0.5, 1))

ggsave("RIdim.pdf", 
       plot = RIdim,
       width = 6,
       height = 3
)

VIdim <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    N == 5000,
    RI > 0.7,
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
  ) %>% 
  ggplot(aes(x = P, y = VI, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab("N° de dimensões.") +
  ylim(c(0,1.2))

ggsave("VIdim.pdf", 
       plot = VIdim,
       width = 6,
       height = 3
)



CPdim <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    N == 5000,
    ClassProp > 0,
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
    #!(P %in% c(5,60,70,80,160))
  ) %>% 
  ggplot(aes(x = P, y = ClassProp, color = Method)) +
  #geom_point(size=2) +
  theme_classic()+
  stat_smooth(method = "lm",
              formula = y ~ poly(x,2), se = FALSE) +
  ylab("Proporção de classificações corretas.")+
  xlab("N° de dimensões.")

ggsave("CPdim.pdf", 
       plot = CPdim,
       width = 6,
       height = 3
)

metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    N == 5000,
    ClassProp > 0,
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
  ) -> dfdim

TIMEdim <- dfdim %>% 
  ggplot(aes(x = P, y = time, color = Method)) +
  theme_classic()+
  ylab("Tempo de execução em segundos.")+
  xlab("N° de dimensões.") +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE)
  #scale_color_manual(values=c("DBSCAN"="black", "Outros"="blue")) +
  #geom_smooth(data=subset(dfdim, Method!="DBSCAN"), method="lm", formula=y~poly(x,2), se=FALSE) +
  #geom_point(data=subset(dfdim, Method=="DBSCAN"), size=2)

ggsave("TIMEdim.pdf", 
       plot = TIMEdim,
       width = 6,
       height = 3
)


################ Nrows Variation  ##############

ARobsDBSCAN <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    P == 5,
    #N <= 200000,
    #AR > 0.6
  ) %>% 
  ggplot(aes(x = N, y = round(AR, 2), color = Method)) +
  geom_point() +
  theme_classic() +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de observações.") +
  xlim(c(1, 200000)) +
  ylim(0.99, 1)
  

ggsave("ARobsDBSCAN.pdf", 
       plot = ARobsDBSCAN,
       width = 6,
       height = 3
)


metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    P == 5,
    #N <= 200000,
    #AR > 0.6,
    Method == "K-Means"
  ) %>% 
  ggplot(aes(x = N, y = AR, color = Method)) +
  geom_point() +
  theme_classic() +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 13), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de observações.")


ggsave("ARobsDBSCAN.pdf", 
       plot = ARobsDBSCAN,
       width = 6,
       height = 3
)


metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    P == 5,
    #N <= 200000,
    AR > 0.6,
    #Method == "DBSCAN"
  ) %>% 
  ggplot(aes(x = N, y = round(AR, 2), color = Method)) +
  geom_line(linewidth=1) +
  theme_classic() +
  stat_smooth(method = "lm",
              formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de observações.")


ggsave("ARobsSC.pdf", 
       plot = ARobsSC,
       width = 6,
       height = 3
)


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

################ Execution Time  ##############






