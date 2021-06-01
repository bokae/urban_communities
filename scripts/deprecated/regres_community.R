### regression for spatial Twitter project ###

# packages
library(dplyr)
library(data.table)
library(plm)
library(stargazer)

setwd(dir = "/home/barcsab/ANET/urban_communities/scripts")

# import data
regdf <- fread("..data/graph_properties.csv", sep=",")

# manipulate variables
regdf$log_income_1 <- log(regdf$income_1) 
regdf$log_income_2 <- log(regdf$income_2)  

# regressions

m0 <- lm(log_income_2 ~ log_income_1, data = regdf)
summary(m0)

m1 <- lm(log_income_2 ~ log_income_1, S, data = regdf)
summary(m1)

m2 <- lm(log_income_2 ~ log_income_1, S, population_1, data = regdf)
summary(m2)

# m3 - add degree

stargazer(m0, m1, m2, out='../outputs/regres_commun_1.html')
stargazer(m0, m1, m2, out='../outputs/regres_commun_1.txt')