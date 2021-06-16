### regression for spatial Twitter project ###
### COMMUNITY MODULARITY REGRESSIONS ###  

# packages
library(dplyr)
library(data.table)
library(plm)
library(stargazer)

#setwd(dir = "home/barcsab/ANET/urban_communities/barcsab/scripts")

# import data
reg_community_df <- fread("../data/community_level_data_2.csv.gz", sep=",")

# regressions

m0 <- lm(mod_S_p ~ income_avg_1 + city, data=reg_community_df)
summary(m0)

m1 <- lm(mod_S_p ~ income_avg_1 + city, data=reg_community_df)
summary(m1)


# regressions
stargazer(m0, m1, out='../outputs/regression_communities.html') 
stargazer(m0, m1, out='../outputs/regression_communities.txt')

