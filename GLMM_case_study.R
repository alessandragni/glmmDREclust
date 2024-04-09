## PARAMETRIC GLMM applied to case study
## Outputs are saved into output/df_level2_pred.csv

#__________________
## Libraries import
library(nlme)
library(lme4)
library(lattice)
library(dplyr)
library(readr)
library(flexmix)

#__________________
## Dataframe import
df <- read_csv("data/df_level2.csv") 

#_________________
## A) POISSON response

# fit the model without fixed intercept
modPoi = glmer(Y_MATH ~ -1 + (1 | CNT) + scale(mean_ESCS_std) + scale(SCHSIZE), 
             data = df,
             family = poisson(link = "log"))

summary(modPoi)

df$pred_GLMM_Poi <- predict(modPoi, df, type="response")

for(i in sort(as.vector(ranef(modPoi)$CNT$`(Intercept)`))){
  print(i)
}


rr1 <- ranef(modPoi)
dd <- as.data.frame(rr1)
if (require(ggplot2)) {
  ggplot(dd, aes(y=grp,x=condval)) +
    geom_point() + facet_wrap(~term,scales="free_x") +
    geom_errorbarh(aes(xmin=condval -2*condsd,
                       xmax=condval +2*condsd), height=0) + 
    theme_bw() + xlab("") + ylab('') +
    ggtitle("Poisson response")
}

ggsave(filename = 'ran_int_Poi.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 4.5,
       height = 6,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)



#_________________
## B) BERNOULLI response

# fit the model without fixed intercept
modBer = glmer(Y_BIN_MATH ~ -1 + (1 | CNT) + scale(SCHSIZE) + scale(mean_ESCS_std), 
             data = df,
             family = binomial(link = "logit"))
summary(modBer)
df$pred_GLMM_Ber <- predict(modBer, df, type="response")


for(i in sort(as.vector(ranef(modBer)$CNT$`(Intercept)`))){
  print(i)
}

  
rr1 <- ranef(modBer)
dd <- as.data.frame(rr1)
if (require(ggplot2)) {
  ggplot(dd, aes(y=grp,x=condval)) +
    geom_point() + facet_wrap(~term,scales="free_x") +
    geom_errorbarh(aes(xmin=condval -2*condsd,
                       xmax=condval +2*condsd), height=0) + 
    theme_bw() + xlab("") + ylab('') +
    ggtitle("Bernoulli response")
}

ggsave(filename = 'ran_int_Bern.pdf',
       plot = last_plot(),
       device = NULL,
       scale = 1,
       width = 5,
       height = 6,
       units = "in",
       dpi = 300,
       limitsize = TRUE,
       bg = NULL)


#_________________
## Save the GLMM predictions in df_level2_pred.csv

write.csv(df,"output/df_level2_pred.csv", row.names = FALSE)




