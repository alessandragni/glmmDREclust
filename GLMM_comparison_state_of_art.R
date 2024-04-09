# import libraries
library(readr)
library(PRROC)
library(dplyr)
library(lattice)
library(stats)
library(lme4)
library(sjPlot)
library(tidyverse)
library(hrbrthemes)
library(viridis)
library(ggplot2)
library(reshape)


########################################################
# Poisson case - random intercept only - 1 fixed slope #
# ran_var = False; ran_int = True
########################################################

# set your working directory here
setwd("output/comparison_state_of_art/Poisson_DG_output")

temp = list.files(pattern="*.csv")
for (i in 1:length(temp)) {
  assign(temp[i], read.csv(temp[i]))
}


#______________________________________________________________________________
# PLOT FOR THE GLMM_10

num = 99
df.list<-lapply(0:num, function(x) eval(parse(text=paste0("Poisson_", x, '.csv')))) 
# In order to store all datasets in one list using their name
names(df.list)<-lapply(0:num, function(x) paste0("Poisson_", x, '.csv')) 
# Adding the name of each df in case you want to unlist the list afterwards

results = matrix(NA, nrow = length(df.list), ncol = 3)
colnames(results) = c('MSE', 'MSE_log', 'Chi-Squared-Error')
random_effects = matrix(data = NA, nrow = length(df.list), ncol = 10)
fix_effect = matrix(data = NA, nrow = length(df.list), ncol = 1)


for (i in 1:length(df.list)) {
  df.list[[i]]$g = as.factor(df.list[[i]]$g)
  
  df.list[[i]]$g_clustered = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'g_clustered'] = 1
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'g_clustered'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'g_clustered'] = 3
  df.list[[i]]$g_clustered = as.factor(df.list[[i]]$g_clustered)
  
  df.list[[i]]$random_int = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'random_int'] = 2.5
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'random_int'] = 1
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'random_int'] = -1
  
  df.list[[i]]$fixed = 0.3
  
  
  glmer1 <- glmer(y ~ -1 + x1 +(1|g), data = df.list[[i]], family = poisson(link = "log"))
  p_predicted1 = predict(glmer1, type = "response")
  y_predicted1 = round(p_predicted1)
  
  #https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
  
  results[i,1] = mean((df.list[[i]]$y - y_predicted1)^2)
  results[i,2] = mean( (log(df.list[[i]]$y + 1) - log(y_predicted1 + 1))^2 )
  results[i,3] = mean((df.list[[i]]$y - y_predicted1)^2/(y_predicted1+1))
  
  random_effects[i,] = as.vector(ranef(glmer1)$g[1])$`(Intercept)`
  fix_effect[i,1] = coef(summary(glmer1))[1]
  
}


df_ran = as.data.frame(random_effects)
df_fix = as.data.frame(fix_effect)
df_results = as.data.frame(results)
colnames(df_ran) = c('b1,1', 'b1,2', 'b1,3', 'b1,4', 'b1,5', 'b1,6', 'b1,7', 'b1,8', 'b1,9', 'b1,10')
colnames(df_fix) = c('beta1')

meltran <- melt(df_ran)
meltfix <- melt(df_fix)
meltres <- melt(df_results)
melted = rbind(meltfix, meltran)

cbp1 <- c("#999999", "#E69F00", "#E69F00", 
          "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9",
          "#009E73", "#009E73", "#009E73")

melted %>% ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) + 
  geom_segment(aes(x = 0.5, xend = 1.5, y = 0.3, yend = 0.3), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 3.5, y = 2.5, yend = 2.5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 8.5, y = 1, yend = 1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 8.5, xend = 11.5, y = -1, yend = -1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
      legend.position="none",
      plot.title = element_text(size=11)
    ) + ggtitle("Coefficients for DGP through GLMM - 10 groups") + xlab("") + ylab("") + theme(plot.title = element_text(hjust = 0.5))
  

ggsave(filename = 'plot_10_glmer_coeff_POI.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 8.5,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)


#_______________________________________________________________________________
# PLOT FOR THE GLMM_3

results2 = matrix(NA, nrow = length(df.list), ncol = 3)
colnames(results2) = c('MSE', 'MSE_log', 'Chi-Squared-Error')
random_effects = matrix(data = NA, nrow = length(df.list), ncol = 3)
fix_effect = matrix(data = NA, nrow = length(df.list), ncol = 1)


for (i in 1:length(df.list)) {
  df.list[[i]]$g = as.factor(df.list[[i]]$g)
  
  df.list[[i]]$g_clustered = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'g_clustered'] = 1
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'g_clustered'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'g_clustered'] = 3
  df.list[[i]]$g_clustered = as.factor(df.list[[i]]$g_clustered)
  
  df.list[[i]]$random_int = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'random_int'] = 2.5
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'random_int'] = 1
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'random_int'] = -1
  
  df.list[[i]]$fixed = 0.3
  
  
  glmer1 <- glmer(y ~ -1 + x1 +(1|g_clustered), data = df.list[[i]], family = poisson(link = "log"))
  p_predicted1 = predict(glmer1, type = "response")
  y_predicted1 = round(p_predicted1)
  
  #https://stats.stackexchange.com/questions/48811/cost-function-for-validating-poisson-regression-models
  
  results2[i,1] = mean((df.list[[i]]$y - y_predicted1)^2)
  results2[i,2] = mean( (log(df.list[[i]]$y + 1) - log(y_predicted1 + 1))^2 )
  results2[i,3] = mean((df.list[[i]]$y - y_predicted1)^2/(y_predicted1+1))
  
  random_effects[i,] = as.vector(ranef(glmer1)$g[1])$`(Intercept)`
  fix_effect[i,1] = coef(summary(glmer1))[1]
  
}


df_ran = as.data.frame(random_effects)
df_fix = as.data.frame(fix_effect)
df_results2 = as.data.frame(results2)
colnames(df_ran) = c('c1,1', 'c1,2', 'c1,3')
colnames(df_fix) = c('beta1')


meltran <- melt(df_ran)
meltfix <- melt(df_fix)
melted = rbind(meltfix, meltran)
meltres2 <- melt(df_results2)


cbp1 <- c("#999999", "#E69F00", 
          "#56B4E9", 
          "#009E73")
melted %>%
  ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) +
  geom_segment(aes(x = 0.5, xend = 1.5, y = 0.3, yend = 0.3), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 2.5, y = 2.5, yend = 2.5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 2.5, xend = 3.5, y = 1, yend = 1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 4.5, y = -1, yend = -1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Coefficients for DGP through GLMM - 3 clusters") +
  xlab("")  + ylab("")


ggsave(filename = 'plot_3_glmer_coeff_POI.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 4.2,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)


#________________________________________________________________________________
# PLOT FOR THE GLMMDRE_3

# set your working directory here
setwd("output/comparison_state_of_art/Poisson_GLMMDRE_output")

melt <- read_csv("melt_POI.csv")
results3 <- read_csv("results3_POI.csv")


cbp1 <- c("#999999", "#E69F00", 
          "#56B4E9", 
          "#009E73")
melt %>%
  ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) +
  geom_segment(aes(x = 0.5, xend = 1.5,  y = 0.3, yend = 0.3), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 2.5,  y = 2.5, yend = 2.5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 2.5, xend = 3.5,  y = 1, yend = 1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 4.5,  y = -1, yend = -1), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Coefficients for DGP through GLMMDRE - 3 clusters") +
  xlab("") + ylab("")


ggsave(filename = 'plot_3_GLMMDRE_coeff_POI.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 4.2,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)




#________________________________________________________________________________
# For the table

# results for 10
# results2 for 3
# results3 for 3 GLMMDRE
apply(results,2,mean)
apply(results2,2,mean)
apply(results3,2,mean)

apply(results,2,sd)
apply(results2,2,sd)
apply(results3,2,sd)

apply(results,2,quantile)
apply(results2,2,quantile)
apply(results3,2,quantile)



########################################################
# Bernoulli case - random intercept only - 1 fixed slope #
# ran_var = False; ran_int = True
########################################################

# set your working directory here
setwd("output/comparison_state_of_art/Bernoulli_DG_output")

temp = list.files(pattern="*.csv")
for (i in 1:length(temp)) {
  assign(temp[i], read.csv(temp[i]))
}


#______________________________________________________________________________
# PLOT FOR THE GLMM_10

num = 99
df.list<-lapply(0:num, function(x) eval(parse(text=paste0("Bernoulli_", x, '.csv')))) #In order to store all datasets in one list using their name
names(df.list)<-lapply(0:num, function(x) paste0("Bernoulli_", x, '.csv')) #Adding the name of each df in case you want to unlist the list afterwards

results = matrix(NA, nrow = length(df.list), ncol = 3)
colnames(results) = c('Sensitivity', 'Specificity', 'Accuracy')
random_effects = matrix(data = NA, nrow = length(df.list), ncol = 10)
fix_effect = matrix(data = NA, nrow = length(df.list), ncol = 1)


for (i in 1:length(df.list)) {
  df.list[[i]]$g = as.factor(df.list[[i]]$g)
  
  df.list[[i]]$g_clustered = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'g_clustered'] = 1
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'g_clustered'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'g_clustered'] = 3
  df.list[[i]]$g_clustered = as.factor(df.list[[i]]$g_clustered)
  
  df.list[[i]]$random_int = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'random_int'] = 5
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'random_int'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'random_int'] = -10
  
  df.list[[i]]$fixed = -6
  
  
  glmer1 <- glmer(y ~ -1 + x1 +(1|g), data = df.list[[i]], family = binomial(link = "logit"))
  p_predicted1 = predict(glmer1, type = "response")
  y_predicted1 = if_else(p_predicted1 > 0.5, 1L, 0L)
  
  t = table(y_predicted1, 'actual' = df.list[[i]]$y)
  TP = t[2,2]
  TN = t[1,1]
  FN = t[1,2]
  FP = t[2,1]
  
  Sensitivity = TP/(TP + FN) 
  Specificity = TN/(TN + FP) 
  Accuracy = (TN + TP)/(TN+TP+FN+FP) 
  
  results[i,1] = Sensitivity
  results[i,2] = Specificity
  results[i,3] = Accuracy
  
  random_effects[i,] = as.vector(ranef(glmer1)$g[1])$`(Intercept)`
  fix_effect[i,1] = coef(summary(glmer1))[1]
  
}


df_ran = as.data.frame(random_effects)
df_fix = as.data.frame(fix_effect)
df_results = as.data.frame(results)
colnames(df_ran) = c('b1,1', 'b1,2', 'b1,3', 'b1,4', 'b1,5', 'b1,6', 'b1,7', 'b1,8', 'b1,9', 'b1,10')
colnames(df_fix) = c('beta1')

meltran <- melt(df_ran)
meltfix <- melt(df_fix)
meltres <- melt(df_results)
melted = rbind(meltfix, meltran)

cbp1 <- c("#999999", "#E69F00", "#E69F00", 
          "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9", "#56B4E9",
          "#009E73", "#009E73", "#009E73")
melted %>% ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) + 
  geom_segment(aes(x = 0.5, xend = 1.5, y = -6, yend = -6), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 3.5, y = 5, yend = 5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 8.5, y = 2, yend = 2), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 8.5, xend = 11.5, y = -10, yend = -10), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
      legend.position="none",
      plot.title = element_text(size=11)
    ) + ggtitle("Coefficients for DGP (i) through GLMM - 10 groups") + xlab("") + ylab("")
  

ggsave(filename = 'plot_10_glmer_coeff.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 7,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)


#_______________________________________________________________________________
# PLOT FOR THE GLMM_3

results2 = matrix(NA, nrow = length(df.list), ncol = 3)
colnames(results2) = c('Sensitivity', 'Specificity', 'Accuracy')
random_effects = matrix(data = NA, nrow = length(df.list), ncol = 3)
fix_effect = matrix(data = NA, nrow = length(df.list), ncol = 1)


for (i in 1:length(df.list)) {
  df.list[[i]]$g = as.factor(df.list[[i]]$g)
  
  df.list[[i]]$g_clustered = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'g_clustered'] = 1
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'g_clustered'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'g_clustered'] = 3
  df.list[[i]]$g_clustered = as.factor(df.list[[i]]$g_clustered)
  
  df.list[[i]]$random_int = NULL
  df.list[[i]][df.list[[i]]$g==1 | df.list[[i]]$g==2, 'random_int'] = 5
  df.list[[i]][df.list[[i]]$g==3 | df.list[[i]]$g==4 | df.list[[i]]$g==5 | df.list[[i]]$g==6 | df.list[[i]]$g==7, 'random_int'] = 2
  df.list[[i]][df.list[[i]]$g==8 | df.list[[i]]$g==9 | df.list[[i]]$g==10, 'random_int'] = -10
  
  df.list[[i]]$fixed = -6
  
  
  glmer1 <- glmer(y ~ -1 + x1 +(1|g_clustered), data = df.list[[i]], family = binomial(link = "logit"))
  p_predicted1 = predict(glmer1, type = "response")
  y_predicted1 = if_else(p_predicted1 > 0.5, 1L, 0L)
  
  t = table(y_predicted1, 'actual' = df.list[[i]]$y)
  TP = t[2,2]
  TN = t[1,1]
  FN = t[1,2]
  FP = t[2,1]
  
  Sensitivity = TP/(TP + FN) 
  Specificity = TN/(TN + FP) 
  Accuracy = (TN + TP)/(TN+TP+FN+FP) 
  
  results2[i,1] = Sensitivity
  results2[i,2] = Specificity
  results2[i,3] = Accuracy
  
  random_effects[i,] = as.vector(ranef(glmer1)$g[1])$`(Intercept)`
  fix_effect[i,1] = coef(summary(glmer1))[1]
  
}


df_ran = as.data.frame(random_effects)
df_fix = as.data.frame(fix_effect)
df_results2 = as.data.frame(results2)
colnames(df_ran) = c('c1,1', 'c1,2', 'c1,3')
colnames(df_fix) = c('beta1')


meltran <- melt(df_ran)
meltfix <- melt(df_fix)
melted = rbind(meltfix, meltran)
meltres2 <- melt(df_results2)


cbp1 <- c("#999999", "#E69F00", 
          "#56B4E9", 
          "#009E73")
melted %>%
  ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) +
  geom_segment(aes(x = 0.5, xend = 1.5, y = -6, yend = -6), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 2.5, y = 5, yend = 5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 2.5, xend = 3.5, y = 2, yend = 2), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 4.5, y = -10, yend = -10), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Coefficients for DGP (i) through GLMM - 3 clusters") +
  xlab("") + ylab("")


ggsave(filename = 'plot_3_glmer_coeff.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 4.2,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)


#________________________________________________________________________________
# PLOT FOR THE GLMMDRE_3

# set your working directory here
setwd("output/comparison_state_of_art/Bernoulli_GLMMDRE_output")

melt <- read_csv("melt.csv")
results3 <- read_csv("results3.csv")


cbp1 <- c("#999999", "#E69F00", 
          "#56B4E9", 
          "#009E73")
melt %>%
  ggplot( aes(x=variable, y=value)) + # fill=name
  geom_boxplot(fill = cbp1) +
  geom_segment(aes(x = 0.5, xend = 1.5, y = -6, yend = -6), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 1.5, xend = 2.5, y = 5, yend = 5), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 2.5, xend = 3.5, y = 2, yend = 2), col = 'red', linewidth = 0.8, linetype = "dashed") +
  geom_segment(aes(x = 3.5, xend = 4.5, y = -10, yend = -10), col = 'red', linewidth = 0.8, linetype = "dashed") +
  scale_fill_viridis(discrete = TRUE, alpha=0.6, option="A") +
  theme_minimal() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("Coefficients for DGP (i) through GLMMDRE - 3 clusters") +
  xlab("") + ylab("")


ggsave(filename = 'plot_3_GLMMDRE_coeff.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 4.2,
       height = 3.5,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)




#________________________________________________________________________________
# For the table

# results for 10
# results2 for 3
# results 3 for 3 GLMMDRE
apply(results,2,mean)
apply(results2,2,mean)
apply(results3,2,mean)

apply(results,2,sd)
apply(results2,2,sd)
apply(results3,2,sd)

apply(results,2,quantile)
apply(results2,2,quantile)
apply(results3,2,quantile)




