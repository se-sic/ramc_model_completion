library(tidyverse)
library(tidyr)
library(dplyr)
library(corrplot)
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}

#For the second experiment, replace experiment1 by experiment2. Change result_columns to: result_columns <- c("score", "score_2")
stats_ada <- read.csv("./../all_results_ada.csv", header=TRUE, sep=";")
stats_curie <- read.csv("./../all_results_curie.csv", header=TRUE, sep=";")
stats_davinci <- read.csv("./../all_results_davinci.csv", header=TRUE, sep=";")

# Combine all results
stats <- rbind(stats_ada, stats_curie, stats_davinci)

result_columns_factorial <- c("Pattern_Correct_1_Rank_factorial", "Pattern_Correct_2_Rank_factorial", "Pattern_Correct_3_Rank_factorial")
result_columns_probability <- c("Pattern_Correct_1_Rank_probability", "Pattern_Correct_2_Rank_probability", "Pattern_Correct_3_Rank_probability")
result_columns_edges_scaled <- c("Pattern_Correct_1_Rank_edges_scaled", "Pattern_Correct_2_Rank_edges_scaled", "Pattern_Correct_3_Rank_edges_scaled")
result_columns_compression <- c("Pattern_Correct_1_Rank_compression", "Pattern_Correct_2_Rank_compression", "Pattern_Correct_3_Rank_compression")
result_columns <- result_columns_compression
#
###################################### Compare the rank of Pattern 1 for models ############################

#stats_correct_1 <- stats %>% select(Id, Pattern_Correct_1_Rank_factorial,Pattern_Correct_2_Rank_factorial, Base_Model, Epochs)
#stats_correct_1 <- pivot_wider(stats_correct_1, names_from = c(Base_Model, Epochs), values_from = c(Pattern_Correct_1_Rank_factorial, Pattern_Correct_2_Rank_factorial))
#mat_temp <-column_to_rownames(stats_correct_1, "Id")

#minusOneToMinusHundred <- function(c) {if (c == -1){
#  c = 50
#} 
#  return(c);
#}

#data_matrix <-as.matrix(mat_temp)
#data_matrix <- apply(data_matrix, 1:2, minusOneToMinusHundred)
#heatmap(data_matrix, Rowv = NA, Colv = NA,  scale = "none", margins=c(20,10))

###################################### Some Method definitions #############################################
### Correlation table for Latex
library(xtable)

corstarsl <- function(x, type=c('pearson')){ 
  require(Hmisc) 
  x <- as.matrix(x) 
  R <- rcorr(x, type=type)$r 
  p <- rcorr(x, type=type)$P 
  
  ## define notions for significance levels; spacing is important.
  mystars <- ifelse(p < .001, "***", ifelse(p < .01, "** ", ifelse(p < .05, "* ", " ")))
  
  ## trunctuate the matrix that holds the correlations to two decimal
  R <- format(round(cbind(rep(-1.11, ncol(x)), R), 2))[,-1] 
  
  ## build a new matrix that includes the correlations with their apropriate stars 
  Rnew <- matrix(paste(R, mystars, sep=""), ncol=ncol(x)) 
  diag(Rnew) <- paste(diag(R), " ", sep="") 
  rownames(Rnew) <- colnames(x) 
  colnames(Rnew) <- paste(colnames(x), "", sep="") 
  
  ## remove upper triangle
  Rnew <- as.matrix(Rnew)
  Rnew[upper.tri(Rnew, diag = TRUE)] <- ""
  Rnew <- as.data.frame(Rnew) 
  
  ## remove last column and return the matrix (which is now a data frame)
  Rnew <- cbind(Rnew[1:length(Rnew)-1])
  return(Rnew) 
}

# Given a list of ranks for the true documents, this function returns the average precision metric
library(comprehenr)
average_precision <- function(rank_list, k){
  relevant_count = length(rank_list)
  # Since we are using -1 for "not found" we have to get rid of these first
  rank_list <- rank_list[rank_list != -1]
  rank_list <- rank_list[rank_list <= k]
  if (length(rank_list) == 0){
    return(0.0);
  }
  # then we sort and compute the precisions
  rank_list <- rank_list[sort.list(rank_list)]
  precs = to_vec(for(i in 1:length(rank_list)) i/rank_list[i])
  
  1 / relevant_count * sum(precs)
}

map_score <- function(rank_lists, k){
  mean(apply(rank_lists, 1, average_precision, k=k))
}

recall_at_k <- function(rank_list, k){
  relevant_count = length(rank_list)
  # Since we are using -1 for "not found" we have to get rid of these first
  rank_list <- rank_list[rank_list != -1]
  rank_list <- rank_list[rank_list <= k]
  if (length(rank_list) == 0){
    return(0.0);
  }
  length(rank_list) / relevant_count
}

mean_recall_at_k <- function(rank_lists, k){
  mean(apply(rank_lists, 1, recall_at_k, k=k))
}

############################ Add some derived values ###############################################
#stats$score[is.na(stats_topn$score)] <- -1
#stats$score_2[is.na(stats_topn$score_2)] <- -1

# relative threshold
#stats <- stats %>% mutate(threshold_relative = support_threshold / cnt_components)

# Selected metric
positions = stats %>% select(result_columns)

stats["average_precision"] <- apply(positions, 1, average_precision, k=max(positions))


# Just the number of correctly identified operations (independent of metric)
stats["count_correct"] <- apply(positions,1,function(x) {sum(x != -1)})


# Factorial metric 
positions = stats %>% select(result_columns_factorial)
stats["average_precision_factorial"] <- apply(positions, 1, average_precision, k=max(positions))

# Edges scaled metric 
positions = stats %>% select(result_columns_edges_scaled)
stats["average_precision_edges_scaled"] <- apply(positions, 1, average_precision, k=max(positions))

# Probability metric 
positions = stats %>% select(result_columns_probability)
stats["average_precision_probability"] <- apply(positions, 1, average_precision, k=max(positions))

# Compression metric 
positions = stats %>% select(result_columns_compression)
stats["average_precision_compression"] <- apply(positions, 1, average_precision, k=max(positions))

## Add numeric correspondance for base model
stats$Base_Model_O <- stats$Base_Model
stats["Base_Model_O"][stats["Base_Model"] == "ada"] <- 0
stats["Base_Model_O"][stats["Base_Model"] == "curie"] <- 1
stats["Base_Model_O"][stats["Base_Model"] == "davinci"] <- 2
stats$Base_Model_O <- as.numeric(stats$Base_Model_O )


stats_inc_davinci <- stats %>% filter(Pertubation == 1.0)

stats <- stats %>% filter("Base_Model" != "davinci")

stats_selected <- stats %>% select(average_precision, Pertubation, EOs, Diffs, Number_Tokens, count_correct, Base_Model)
###################################



#########  Some correlation computations ##################################
# The correlations for ada and curie
stats_temp <- stats %>% select(Base_Model_O, Pertubation,  EOs,  Epochs, count_correct)
cors <- cor(stats_temp, method="spearman")
plot_cor(cors)
corstarsl(stats_temp, type=c("spearman"))

stats_temp <- stats_inc_davinci %>% select(Base_Model_O, Epochs, Diffs, EOs,  Epochs, count_correct)
cors <- cor(stats_temp, method="spearman")
plot_cor(cors)
corstarsl(stats_temp, type=c("spearman"))

stats_temp <- stats %>% filter(Diffs==20, EOs==81) %>% select(Base_Model_O, Epochs, Pertubation, Epochs, count_correct)
cors <- cor(stats_temp, method="spearman")
plot_cor(cors)
corstarsl(stats_temp, type=c("spearman"))

stats_temp <- stats %>% filter(Base_Model == "ada") %>% select(Number_Tokens, EOs, Diffs, Epochs, count_correct)
cors <- cor(stats_temp, method="spearman")
plot_cor(cors)
corstarsl(stats_temp, type=c("spearman"))
############## Some basic stats ##############

ds_ada <- stats %>% filter(Base_Model == 'ada')
ds_curie <- stats %>% filter(Base_Model == 'curie')
ds_davinci <- stats %>% filter(Base_Model == 'davinci')


average_correct_ada = sum(ds_ada$count_correct)/count(ds_ada)
average_correct_curie = sum(ds_curie$count_correct)/count(ds_curie)
average_correct_davinci = sum(ds_davinci$count_correct)/count(ds_davinci)
print(average_correct_ada)
print(average_correct_curie)
print(average_correct_davinci)

ada_difficult <- ds_ada %>% filter(Pertubation == 1.0)
curie_difficult <- ds_curie  %>% filter(Pertubation == 1.0)
average_correct_ada_difficult= sum(ada_difficult$count_correct)/count(ada_difficult)
average_correct_curie_difficult = sum(curie_difficult$count_correct)/count(curie_difficult)
print(average_correct_ada_difficult)
print(average_correct_curie_difficult)

# Training Cost
average_cost_ada = sum(ds_ada$Cost)/count(ds_ada)
average_cost_curie = sum(ds_curie$Cost)/count(ds_curie)
average_cost_davinci = sum(ds_davinci$Cost)/count(ds_davinci)
print(average_cost_ada)
print(average_cost_curie)
print(average_cost_davinci)
total_cost_training = sum(ds_ada$Cost) + sum(ds_curie$Cost) + sum(ds_davinci$Cost)

# Generation Cost
price_ada = 0.0000016
price_curie = 0.000012
price_davinci = 0.00012

average_g_cost_ada = sum(ds_ada$token_count) * price_ada /count(ds_ada)
average_g_cost_curie = sum(ds_curie$token_count) * price_curie /count(ds_curie)
average_g_cost_davinci = sum(ds_davinci$token_count) * price_davinci/count(ds_davinci)
print(average_g_cost_ada)
print(average_g_cost_curie)
print(average_g_cost_davinci)
