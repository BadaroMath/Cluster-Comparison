

library(latex2exp)
library(ggplot2)
library(dplyr)
library(readr)

setwd("F:/TG2/results/Graficos")
metric <- read_csv("F:/TG2/results/resultados_metricas/total_metrics_final.txt",
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
  geom_point(size=1, shape = 21) +
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
  mutate(Peso = ifelse(ClassProp >= .9, 100, 1)) %>%
  ggplot(aes(x = BarOmega, y = ClassProp, color = Method)) +
  geom_point(size=1, shape = 21) +
  theme_classic()+
  geom_smooth(aes(weight=Peso), span = 0.4, se = FALSE) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Proporção de classificações corretas.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$.")) +
  ylim(c(0,1))

ggsave("CPomega.pdf", 
       plot = CPomega,
       width = 6,
       height = 3
)





DIomega <- metrics %>% 
  filter(K == 3, P == 5, N == 5000,
         DI < .05) %>%
  ggplot(aes(x = BarOmega, y = DI, color = Method)) +
  geom_point(size=1, shape = 21) +
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
  geom_point(size=1, shape = 21) +
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
  filter(K == 3, P == 5, N == 5000, VI < 2.5) %>%
  mutate(Peso = ifelse(VI <= 0.1, 100, 1)) %>%
  mutate(Peso = ifelse(VI > 1.5, 10, Peso)) %>%
  ggplot(aes(x = BarOmega, y = VI, color = Method)) +
  geom_point(size=1, shape = 21) +
  theme_classic()+
  geom_smooth(aes(weight=Peso), span = 0.5, se = FALSE) +
  #stat_smooth(aes(weight=Peso),method = "lm",
  #            formula = y ~ poly(x, 5), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("VIomega.pdf", 
       plot = VIomega,
       width = 6,
       height = 3
)


metrics %>% 
  filter(K == 3, P == 5, N == 5000) -> dfomg

TIMEomega <- dfomg %>% ggplot(aes(x = BarOmega, y = time, color = Method)) +
  geom_point(data = subset(dfomg, Method == "Spectral"), size=2) +
  theme_classic()+
  stat_smooth(data = subset(dfomg, Method != "Spectral"),method = "lm",
              formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Tempo de execução em segundos.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("TIMEomega.pdf", 
       plot = TIMEomega,
       width = 6,
       height = 3
)

TIMEspec <- dfomg %>% 
  filter(Method != "Spectral") %>%
  ggplot(aes(x = BarOmega, y = time, color = Method)) +
  #geom_point(data = subset(dfomg, Method == "Spectral"), size=2) +
  geom_point(size=1, shape=21) +
  theme_classic()+
  geom_smooth(method = "loess",span = 0.99999, se = FALSE) +
  #stat_smooth(data = subset(dfomg, Method != "Spectral"),method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Tempo de execução em segundos.")+
  xlab(latex2exp::TeX("$\\bar{\\omega}$."))

ggsave("TIMEspec.pdf", 
       plot = TIMEspec,
       width = 6,
       height = 3
)



################ Clusters Variation  ##############
dfcluster <-  metrics %>% 
  filter(
    BarOmega == 0,
    P == 3,
    N == 5000
  ) %>%
  mutate(Peso = ifelse(AR > 0.99 & Method == "Spectral", 100, 1))%>%
  mutate(Peso = ifelse(K == 11 & Method == "Spectral", 100, Peso))



ARcluster <- dfcluster %>% 
  ggplot(aes(x = K, y = AR, color = Method, weight=Peso)) +
  geom_point(size=1, shape = 21) +
  theme_classic()+
  #stat_smooth(method = "lm",formula = y ~ poly(x, 4), se = FALSE) +
  geom_smooth(data = subset(dfcluster, K <= 11 & Method == "Spectral"),span = 1, se = FALSE, method = "loess") +
  geom_smooth(data = subset(dfcluster, K >= 11 & Method == "Spectral"),span = 0.99999, se = FALSE, method = "loess") +
  geom_smooth(data = subset(dfcluster, Method != "Spectral"),span = 0.99999, se = FALSE, method = "loess") +
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
  geom_point(size=1, shape = 21) +
  theme_classic()+
  geom_smooth(span = 0.9, se = FALSE, method = "loess") +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
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
  geom_point(size=1, shape = 21) +
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
    P %in% c(seq(2,30, 2), seq(30, 200, 10))
    #!(P %in% c(60,70,80,160))
  ) %>% 
  mutate(AR = case_when(Method == "Gaussian Mixture" & AR < 0.7 ~ NaN,
                        TRUE ~ AR),
         Peso = ifelse(AR > 0.95, 100, 1)) %>%
  ggplot(aes(x = P, y = AR, color = Method, weight = Peso)) +
  geom_point(size=1, shape = 21) +
  theme_classic()+
  geom_smooth(method = "loess",span = 0.8, se = FALSE) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de dimensões.")


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
    #RI > 0.7,
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
  ) %>% 
  mutate(VI = case_when(Method == "Gaussian Mixture" & VI > 0.5 ~ NaN,
                        TRUE ~ VI)) %>%  
  ggplot(aes(x = P, y = VI, color = Method)) +
  geom_point(size=1, shape = 21) +
  theme_classic()+
  geom_smooth(method = "loess",span = 0.8, se = FALSE) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab("N° de dimensões.")

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
    P %in% c(seq(2,30, 2), seq(30, 200, 10)),
    #!(P %in% c(5,60,70,80,160))
  ) %>% 
  ggplot(aes(x = P, y = ClassProp, color = Method)) +
  geom_point(size=2) +
  theme_classic()+
  geom_smooth(method = "loess",span = 0.8, se = FALSE) +
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
  geom_smooth(method = "loess",span = 0.9, se = FALSE) +
  geom_point(size = 1, shape = 21)
  #scale_color_manual(values=c("DBSCAN"="black", "Outros"="blue")) +
  #geom_smooth(data=subset(dfdim, Method!="DBSCAN"), method="lm", formula=y~poly(x,2), se=FALSE) +
  #geom_point(data=subset(dfdim, Method=="DBSCAN"), size=2)

ggsave("TIMEdim.pdf", 
       plot = TIMEdim,
       width = 6,
       height = 3
)


################ Nrows Variation  ##############

dfnrow <- metrics %>% 
  filter(
    BarOmega == 0,
    K == 3,
    P == 5,
    #N <= 200000,
    N %in% c(seq(100, 5000, 100), seq(5000, 100000, 10000), seq(100000, 2000000, 100000))
  ) %>%
  mutate(Peso = ifelse(AR>0.9, 100, 1))

ARobsDensity <- dfnrow %>% 
  filter(!(Method %in% c("DBSCAN", "Spectral")),
         AR > 0.6) %>%
  ggplot(aes(x = N, y = round(AR, 2), color = Method)) +
  geom_smooth(method = "loess",span = 1, se = FALSE) +
  geom_point(size = 2) +
  theme_classic() +
  #geom_point(data = subset(dfnrow, Method == "Spectral")) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  xlab("N° de observações.")


ggsave("ARobsDensity.pdf", 
       plot = ARobsDensity,
       width = 6,
       height = 3
)


ARobsDensity <- dfnrow %>% 
  filter(Method %in% c("DBSCAN", "Spectral"),
         AR > 0.6) %>%
  ggplot(aes(x = N, y = round(AR, 2), color = Method, shape = Method)) +
  geom_point(size = 2) +
  theme_classic() +
  #geom_point(data = subset(dfnrow, Method == "Spectral")) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Índice de Rand Ajustado.")+
  scale_shape_manual(values = c(1, 3)) +
  xlab("N° de observações.")
  

ARobsDensity <- dfnrow %>% 
  group_by(Method) %>%
  summarize(max_n = max(N)) %>%
  ggplot(aes(y = Method, x = max_n, fill = Method)) +
  geom_col() +
  theme_classic() +
  geom_vline(xintercept = 100000, linetype = "dashed", color = "red") +
  annotate("text", x = 120000, y = "Spectral", label = "N > 100k: Memory Error", color = "red", hjust = 0) +
  #geom_point(data = subset(dfnrow, Method == "Spectral")) +
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 1), se = FALSE) +
  ylab("Métodos.")+
  xlab("N° de observações.")
  

ggsave("ARobsDensity.pdf", 
       plot = ARobsDensity,
       width = 6,
       height = 3
)



VIN <- dfnrow %>% 
  filter(VI < 0.4) %>%
  ggplot(aes(x = N, y = VI, color = Method)) +
  geom_point(size=2) +
  theme_classic()+
  #stat_smooth(method = "lm",
  #            formula = y ~ poly(x, 2), se = FALSE) +
  ylab("Variação da Informação.")+
  xlab("N° de Observações.")

ggsave("VIN.pdf", 
       plot = VIN,
       width = 6,
       height = 3
)



################ SimDataset example ##############

set.seed(12)
Q <- MixSim(BarOmega = 0.05, K = 4, p = 2)
A <- simdataset(n = 5000, Pi = Q$Pi, Mu = Q$Mu, S = Q$S)

colors <- c("red", "blue", "orange", "purple")

simex <- A |> 
  as.data.frame() |>
  mutate(id = as.factor(id)) |>
  ggplot(aes(x = X.1, y = X.2, color = id)) +
  theme_classic()+
  ylab("Variável 2.")+
  xlab("Variável 1.") +
  geom_point(size = 1, shape = 21) +
  scale_color_manual(values = colors,name = "Agrupamento:")


ggsave("simex.pdf", 
       plot = simex,
       width = 6,
       height = 3
)

################ Conclusões preliminares  ##############
library(xtable)

arbox <- metrics |>
  ggplot(aes(x = Method, y = AR, color = Method)) +
  geom_boxplot() +
  theme_classic() +
  ylab("Índice de Rand Ajustado.")+
  xlab("Métodos.")


ggsave("arbox.pdf", 
       plot = arbox,
       width = 8,
       height = 4
)

artable <- metrics %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(AR), 2),
            `1° Quartil`  = round(quantile(AR, 0.25), 2),
            Mediana       = round(median(AR), 2),
            `3° Quartil`  = round(quantile(AR, 0.75), 2),
            `Max.`        = round(max(AR), 2),
            Média         = round(mean(AR), 2),
            SD            = round(sd(AR), 2)
  )
xtable(artable, caption = "Medidas resumo para o Índice de Rand Ajustado por método utilizado em todos os cenários simulados.")

cpbox <- metrics %>% 
  filter(ClassProp < 1500) %>%
  ggplot(aes(x = Method, y = ClassProp, color = Method)) +
  geom_boxplot() +
  theme_classic() +
  ylab("Proporção de Classificações Corretas.")+
  xlab("Métodos.")

ggsave("cpbox.pdf", 
       plot = cpbox,
       width = 8,
       height = 4
)

cptable <- metrics %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(ClassProp), 2),
            `1° Quartil`  = round(quantile(ClassProp, 0.25), 2),
            Mediana       = round(median(ClassProp), 2),
            `3° Quartil`  = round(quantile(ClassProp, 0.75), 2),
            `Max.`        = round(max(ClassProp), 2),
            Média         = round(mean(ClassProp), 2),
            SD            = round(sd(ClassProp), 2)
  )
xtable(cptable, caption = "Medidas resumo para a Proporção de Classificações Corretas por método utilizado em todos os cenários simulados.")


timebox <- metrics %>%
  ggplot(aes(x = Method, y = time, color = Method)) +
  geom_boxplot() +
  theme_classic()+ 
  ylab("Tempo de execução em segundos.")+
  xlab("Métodos.")

ggsave("timebox.pdf", 
       plot = timebox,
       width = 8,
       height = 4
)


timetable <- metrics %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(time), 2),
            `1° Quartil`  = round(quantile(time, 0.25), 2),
            Mediana       = round(median(time), 2),
            `3° Quartil`  = round(quantile(time, 0.75), 2),
            `Max.`        = round(max(time), 2),
            Média         = round(mean(time), 2),
            SD            = round(sd(time), 2)
  )
xtable(timetable, caption = "Medidas resumo para o tempo de execução em segundos por método utilizado em todos os cenários simulados.")


vibox <- metrics %>%
  ggplot(aes(x = Method, y = VI, color = Method)) +
  geom_boxplot() +
  theme_classic() +
  ylab("Variação da Informação.")+
  xlab("Métodos.")

ggsave("vibox.pdf", 
       plot = vibox,
       width = 8,
       height = 4
)

vitable <- metrics %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(VI), 2),
            `1° Quartil`  = round(quantile(VI, 0.25), 2),
            Mediana       = round(median(VI), 2),
            `3° Quartil`  = round(quantile(VI, 0.75), 2),
            `Max.`        = round(max(VI), 2),
            Média         = round(mean(VI), 2),
            SD            = round(sd(VI), 2)
  )
xtable(vitable, caption = "Medidas resumo para a Variação da Informação por método utilizado em todos os cenários simulados.")



dibox <- metrics %>%
  ggplot(aes(x = Method, y = DI, color = Method)) +
  geom_boxplot() +
  theme_classic() +
  ylab("Índice de Dunn.")+
  xlab("Métodos.")

ggsave("dibox.pdf", 
       plot = dibox,
       width = 8,
       height = 4
)


ditable <- metrics %>%
  #na.omit() %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(DI, na.rm = TRUE), 2),
            `1° Quartil`  = round(quantile(DI, 0.25,na.rm = TRUE), 2),
            Mediana       = round(median(DI, na.rm = TRUE), 2),
            `3° Quartil`  = round(quantile(DI, 0.75, na.rm = TRUE), 2),
            `Max.`        = round(max(DI, na.rm = TRUE), 2),
            Média         = round(mean(DI, na.rm = TRUE), 2),
            SD            = round(sd(DI, na.rm = TRUE), 2)
  )
xtable(ditable, caption = "Medidas resumo para o Índice de Dunn por método utilizado em todos os cenários simulados.")


sibox <- metrics %>%
  ggplot(aes(x = Method, y = SI, color = Method)) +
  geom_boxplot() +
  theme_classic() +
  ylab("Coeficiente de Silhueta.")+
  xlab("Métodos.")



ggsave("sibox.pdf", 
       plot = sibox,
       width = 8,
       height = 4
)


sitable <- metrics %>%
  group_by(Method) %>%
  summarize(`Min.`        = round(min(DI, na.rm = TRUE), 2),
            `1° Quartil`  = round(quantile(DI, 0.25,na.rm = TRUE), 2),
            Mediana       = round(median(DI, na.rm = TRUE), 2),
            `3° Quartil`  = round(quantile(DI, 0.75, na.rm = TRUE), 2),
            `Max.`        = round(max(DI, na.rm = TRUE), 2),
            Média         = round(mean(DI, na.rm = TRUE), 2),
            SD            = round(sd(DI, na.rm = TRUE), 2)
  )
xtable(sitable, caption = "Medidas resumo para o Coeficiente de Silhueta por método utilizado em todos os cenários simulados.")






