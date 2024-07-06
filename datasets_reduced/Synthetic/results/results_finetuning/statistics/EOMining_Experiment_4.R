library(tidyverse)
library(tidyr)
library(dplyr)
library(corrplot)
if (!require("RColorBrewer")) {
  install.packages("RColorBrewer")
  library(RColorBrewer)
}

###  table for Latex
library(xtable)

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


stats_all <- stats
stats_inc_davinci <- stats %>% filter(Pertubation == 1.0)
stats <- stats %>% filter("Base_Model" != "davinci")
###################################

##################################### MAP score /MR@k #############################
##################
compute_MAP <- function(result_list, result_columns, min_k=3){
  result_columns <- c(result_columns)
  max_rank = max(result_list %>% select(result_columns))
  # MAP
  map1 = map_score(result_list %>% select(result_columns), min_k)
  map5 = map_score(result_list %>% select(result_columns), 5)
  map10 = map_score(result_list %>% select(result_columns), 10)
  mapinfty = map_score(result_list %>% select(result_columns), max_rank)
  
  map_matrix = matrix(c(map1, map5, map10, mapinfty), nrow=4, ncol=1)
  rownames(map_matrix) <- c("MAP@3", "MAP@5", "MAP@10", "MAP@Infty")
  
  return(map_matrix)
}


################

# We use -1 for it does not appear at all
rank_topn <- stats_all

#result_columns_factorial <- c("Pattern_Correct_1_Rank_factorial", "Pattern_Correct_2_Rank_factorial", "Pattern_Correct_3_Rank_factorial")
#result_columns_probability <- c("Pattern_Correct_1_Rank_probability", "Pattern_Correct_2_Rank_probability", "Pattern_Correct_3_Rank_probability")
#result_columns_edges_scaled <- c("Pattern_Correct_1_Rank_edges_scaled", "Pattern_Correct_2_Rank_edges_scaled", "Pattern_Correct_3_Rank_edges_scaled")
#result_columns_compression <- c("Pattern_Correct_1_Rank_compression", "Pattern_Correct_2_Rank_compression", "Pattern_Correct_3_Rank_compression")
result_columns <- matrix(c(result_columns_compression, result_columns_factorial, result_columns_probability, result_columns_edges_scaled), nrow=3, ncol=4)
colnames(result_columns) <- c("Compression", "Factorial", "Probability", "Edges Scaled")

map_scores = apply(result_columns, 2, compute_MAP, result_list=rank_topn)
rownames(map_scores) <- c("MAP@3", "MAP@5", "MAP@10", "MAP@Infty")
xtable(map_scores)


# TODO also move this to extra method
#MR@k
#mean_recall_at_k(rank_topn %>% select(result_columns), 1)
#mean_recall_at_k(rank_topn %>% select(result_columns), 2)
#mean_recall_at_k(rank_topn %>% select(result_columns), 5)
#mean_recall_at_k(rank_topn %>% select(result_columns), 10)
#mean_recall_at_k(rank_topn %>% select(result_columns), max_rank)
################# AP correlations ######
ap_matrix = stats %>% select(average_precision_factorial, average_precision_probability, average_precision_edges_scaled, average_precision_compression)
corstarsl(ap_matrix, type="spearman")
#cor_mat <- cor(ap_matrix, method="pearson")
#col<- colorRampPalette(c("blue", "white", "red"))(20)
#corrplot(cor_mat, addCoef.col = "black", # Add coefficient of correlation
#         tl.col="black", tl.srt=45, #Text label color and rotation
#         type="upper", order="hclust", diag=FALSE, col=col)


##################################### Average precision heatmap ################################################
# create a matrix out of the data for heatmap plot
mat_temp <- pivot_wider(stats_all  %>% select(Pertubation, EOs, average_precision_compression), names_from = EOs, values_from = average_precision_compression)
mat_temp <-column_to_rownames(mat_temp, "Pertubation")

#mat_temp <- arrange(mat_temp, EOs, Pertubation)

#mat_temp <- as.numeric(mat_temp)

# average all the AP per EOs and Pertubation
mat_temp <- apply(mat_temp, c(1,2), function(l) mean(sapply(l, as.numeric)))

# sort for the column and row names
mat_temp <- mat_temp[ order(as.numeric(row.names(mat_temp))), ]
mat_temp <- mat_temp[ , order(as.numeric(colnames(mat_temp)))]

data_matrix <- as.matrix(mat_temp)

colMain <- colorRampPalette(brewer.pal(8, "Greys"))(25)
heatmap(data_matrix,  Rowv = NA, Colv = NA,  scale = "none",col=colMain)
# How to get the right numbers fo the legend here????
#legend(x="bottomright", legend=c("0.17", "0.5", "0.5"), 
#       fill=colorRampPalette(brewer.pal(8, "Blues"))(3))
