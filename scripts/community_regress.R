### regression for spatial Twitter project ###
### COMMUNITY MODULARITY REGRESSIONS ###  

# packages
library(dplyr)
library(data.table)
library(plm)
library(stargazer)

setwd("/home/barcsab/urban_communities/data")

# import data
reg_community_df <- fread("../data/community_level_data_2.csv.gz", sep=",")

# regressions



'educ_ba_p_diff_avg_1', 'white_p_diff_avg_1', 'black_p_diff_avg_1',
       'native_p_diff_avg_1', 'asian_p_diff_avg_1', 'income_diff_avg_1'


m00 <- lm(mod_S_p ~ tract_sum + income_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m00)

m01 <- lm(mod_S_p ~ tract_sum + educ_ba_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m01)

m02 <- lm(mod_S_p ~ tract_sum + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m02)

m03 <- lm(mod_S_p ~ tract_sum + income_diff_avg_1 + educ_ba_p_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m03)



m10 <- lm(mod_S_p ~ tract_sum + income_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m10)

m11 <- lm(mod_S_p ~ tract_sum + educ_ba_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m11)

m12 <- lm(mod_S_p ~ tract_sum + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m12)

m13 <- lm(mod_S_p ~ tract_sum + income_diff_avg_1 + educ_ba_p_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m13)



# averages

m20 <- lm(mod_S_p ~ tract_sum + income_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m20)

m21 <- lm(mod_S_p ~ tract_sum + educ_ba_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m21)

m22 <- lm(mod_S_p ~ tract_sum + black_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m22)

m23 <- lm(mod_S_p ~ tract_sum + income_avg_1 + educ_ba_avg_1 + black_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m23)



m30 <- lm(mod_S_p ~ tract_sum + income_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m30)

m31 <- lm(mod_S_p ~ tract_sum + educ_ba_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m31)

m32 <- lm(mod_S_p ~ tract_sum + black_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m32)

m33 <- lm(mod_S_p ~ tract_sum + income_avg_1 + educ_ba_avg_1 + black_avg_1 + as.factor(city), data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m33)


############################################################################
# SECTION 2
# mod_S_p not an independent variable explains demographic variable

# p_diff
# mgn p_diff
m40 <- lm(income_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m40)

m41 <- lm(educ_ba_p_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m41)

m42 <- lm(black_p_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m42)

m43 <- lm(income_diff_avg_1 ~ tract_sum + mod_S_p + educ_ba_p_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m43)

m44 <- lm(educ_ba_p_diff_avg_1 ~ tract_sum + mod_S_p + income_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m44)

m45 <- lm(black_p_diff_avg_1 ~ tract_sum + mod_S_p + income_diff_avg_1 + educ_ba_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m45)


# ms p_diff
m50 <- lm(income_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m50)

m51 <- lm(educ_ba_p_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m51)

m52 <- lm(black_p_diff_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m52)

m53 <- lm(income_diff_avg_1 ~ tract_sum + mod_S_p + educ_ba_p_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m53)

m54 <- lm(educ_ba_p_diff_avg_1 ~ tract_sum + mod_S_p + income_diff_avg_1 + black_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m54)

m55 <- lm(black_p_diff_avg_1 ~ tract_sum + mod_S_p + income_diff_avg_1 + educ_ba_p_diff_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m55)



# average
# mgn avg
m60 <- lm(income_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m60)

m61 <- lm(educ_ba_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m61)

m62 <- lm(black_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m62)

m63 <- lm(income_avg_1 ~ tract_sum + mod_S_p + educ_ba_avg_1 + black_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m63)

m64 <- lm(educ_ba_avg_1 ~ tract_sum + mod_S_p + income_avg_1 + black_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m64)

m65 <- lm(black_avg_1 ~ tract_sum + mod_S_p + income_avg_1 + educ_ba_avg_1, data=subset(reg_community_df, algorithm_type == "mgn" & g_type == 'fol_hh'))
summary(m65)


# ms avg
m70 <- lm(income_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m70)

m71 <- lm(educ_ba_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m71)

m72 <- lm(black_avg_1 ~ tract_sum + mod_S_p, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m72)

m73 <- lm(income_avg_1 ~ tract_sum + mod_S_p + educ_ba_avg_1 + black_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m73)

m74 <- lm(educ_ba_avg_1 ~ tract_sum + mod_S_p + income_avg_1 + black_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m74)

m75 <- lm(black_avg_1 ~ tract_sum + mod_S_p + income_avg_1 + educ_ba_avg_1, data=subset(reg_community_df, algorithm_type == "ms" & g_type == 'fol_hh'))
summary(m75)





# regressions
stargazer(m00, m01, m02, m03, m10, m11, m12, m13, digits=2, digits.extra=6, out='../outputs/regression_communities.html')
stargazer(m20, m21, m22, m23, m30, m31, m32, m33, digits=2, digits.extra=6, out='../outputs/regression_communities_2.html') 
#stargazer(m00, m01, m02, m03, m10, m11, m12, m13, out='../outputs/regression_communities.txt')

