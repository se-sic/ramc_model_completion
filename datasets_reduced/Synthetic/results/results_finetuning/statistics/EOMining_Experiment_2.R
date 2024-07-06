library(tidyverse)
library(tidyr)
library(dplyr)
library(corrplot)
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}
require(data.table) ## 1.9.3 commit 1267

# Load results csv for all evaluated models
# Note that doing the analysis for all repositories can be quite costly, we selected specific models
# Ada, 6 epochs, D=10, EO=31, p=0.0
data_ada = read.csv("./../results_completion_ada_6_D10E31p0.csv", header = TRUE, sep=";")
# Curie, 6 epochs, D=20, EO=51, p=0.0
data_curie = read.csv("./../results_completion_curie_6_D20E51p0.csv", header = TRUE, sep=";")

#For the second experiment, replace experiment1 by experiment2. Change result_columns to: result_columns <- c("score", "score_2")
stats_ada <- read.csv("./../all_results_ada.csv", header=TRUE, sep=";")
stats_curie <- read.csv("./../all_results_curie.csv", header=TRUE, sep=";")
stats_davinci <- read.csv("./../all_results_davinci.csv", header=TRUE, sep=";")

# Combine all results
eo_data_combined <- rbind(stats_ada, stats_curie, stats_davinci)

# Combine all results
data_combined <- bind_rows(data_ada, data_curie)

# TODO remove this again
#data$missing_edges <- data$total_edges - data$prompt_edges 
#data$best_rank <- ifelse(data$completion_best_rank_result==data$completion_best_eval_result,0,data$nb_completion_candidates)
#write_excel_csv2(data, "./../results_completion_ada_6_D10E31p0.csv", append = FALSE, col_names = TRUE, )

# Select dataset
data <- data_combined

########################################################################################
# Utils
plot_cor <- function(cor_mat){
  col<- colorRampPalette(c("blue", "white", "red"))(20)
  corrplot(cor_mat, addCoef.col = "black", # Add coefficient of correlation
           tl.col="black", tl.srt=45, #Text label color and rotation
           type="upper", order="hclust", diag=FALSE, col=col)
}


########################################################################################

########################### Correlations ################################################
colnames(data)
#compute correlations
stats_temp <- data %>% select(missing_edges, total_edges, nb_completion_candidates, completion_best_rank_score, completion_best_eval_score) 
cors <- cor(stats_temp, method="spearman")
plot_cor(cors)
corstarsl(stats_temp, type=c("spearman"))
########################################################################################

########################## Basic stats ##############################################

#Average number of missing lines, edges in ground truth,... (i.e., difficulty of the completion assessment)
average_missing = sum(data$missing_edges)/count(data)
average_edges = sum(data$total_edges)/count(data)
average_nb_candidates = sum(data$nb_completion_candidates)/count(data)


# Correct completions
correct_data = data %>% filter(completion_best_eval_result == "ISOMORPHIC")
average_correct = count(correct_data)/count(data)
average_rank = sum(correct_data$best_rank)/count(correct_data)

# As a sanity check if our dataset generation works, we plot the ratio of prompt vs. total edges
prompt_vs_total = data$prompt_edges / data$total_edges
# Our generation should yield a 3-modal distribution, we use Ameijeiras-Alonso et al. (2019) excess mass test to verify 3-modality
install.packages('multimode')
library(multimode)
modetest(prompt_vs_total, mod0=2, method="ACR")

hist(prompt_vs_total)


# Compute the statistics for the average token accuracy over all repositories
summary(eo_data_combined$Average_Token_Acc)



