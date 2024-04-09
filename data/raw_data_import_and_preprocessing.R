## 1. DATA IMPORT 

#__________________
## Libraries import
library(foreign)
library(dplyr)


#_____________________________________________
## 1.A) Schools questionnaire data file import

#_________________
## Raw Data import
## downloaded from  https://webfs.oecd.org/pisa2018/SPSS_SCH_QQQ.zip
SCH = read.spss('raw_data_SCH.sav', reencode='utf-8')

sch = data.frame('CNT' = SCH[["CNT"]],
                 'CNTSCHID' = SCH[["CNTSCHID"]],
                 'PRIVATESCH' = SCH[["PRIVATESCH"]],
                 'STRATIO' = as.numeric(as.character(SCH[["STRATIO"]])),
                 'SCHSIZE' = as.numeric(as.character(SCH[["SCHSIZE"]]))
                 )

sch = sch %>% filter(PRIVATESCH != 'invalid' & 
                       PRIVATESCH != "       " & PRIVATESCH != 'missing')
sch = na.omit(sch)

sch$PRIVATESCH[sch$PRIVATESCH == "PRIVATE"] <- "private"
sch$PRIVATESCH[sch$PRIVATESCH == "PUBLIC"] <- "public"
sch$PRIVATESCH[sch$PRIVATESCH == "PUBLIC "] <- "public"
sch$PRIVATESCH[sch$PRIVATESCH == "public "] <- "public"


#_____________________________________________
## 1.B) Students questionnaire data file import

#_____________
## Raw Data import
## downloaded from https://webfs.oecd.org/pisa2018/SPSS_STU_QQQ.zip
STU = read.spss('raw_data_STU.sav', reencode='utf-8')

stu = data.frame('CNT' = STU[["CNT"]],
                 'CNTSCHID' = STU[["CNTSCHID"]],
                 'PV1MATH' = as.numeric(as.character(STU[["PV1MATH"]])),
                 'ESCS' = as.numeric(as.character(STU[["ESCS"]])) )

stu = na.omit(stu)


# https://www.oecd.org/pisa/pisa-for-development/pisafordevelopment2018technicalreport/PISA-D%20TR%20Chapter%2015%20-%20Proficiency%20Scale%20Construction%20-%20final.pdf
stu$MATHbelow = ifelse(stu$PV1MATH <= 482.38, 1, 0) # low achieving students definition

by_sch = stu %>% group_by(CNTSCHID)

schools = by_sch %>% summarise(
  sum_MATHbelow = sum(MATHbelow),
  mean_ESCS = mean(ESCS)
)



#___________________________________________
## 2) CREATION OF THE PREPROCESSED DATAFRAME

df <- merge(x = sch, y=schools, 
            by = 'CNTSCHID', all.x=TRUE)
df = na.omit(df)
df

df$CNT = as.factor(as.character(as.factor(df$CNT)))
df$PRIVATESCH = as.factor(df$PRIVATESCH)

# restrict to schools with more than 10 students
df = df %>% filter(SCHSIZE > 10)

# scale mean_ESCS with respect to the country CNT
boxplot(df$mean_ESCS ~ df$CNT)
df <- within(df, mean_ESCS_std <- ave(mean_ESCS, CNT, FUN=function(x) (scale(x))))
boxplot(df$mean_ESCS_std ~ df$CNT)
boxplot(df$mean_ESCS ~ df$PRIVATESCH)

# compute Y_MATH as the rounded percentage of low-achieving students
df$Y_MATH = round(100 * df$sum_MATHbelow / df$SCHSIZE)

# compute Y_BIN_MATH
df$Y_BIN_MATH = ifelse(df$Y_MATH > 2, 1, 0)
table(df$Y_BIN_MATH)

# save as a .csv
write.csv(df,"df_level2.csv", row.names = FALSE)



