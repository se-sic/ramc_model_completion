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


############################# Utils ######################################################

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

###########################################################################################

####################### Load dataset #####################################################

#For the second experiment, replace experiment1 by experiment2. Change result_columns to: result_columns <- c("score", "score_2")
stats_ada <- read.csv("./../all_results_ada.csv", header=TRUE, sep=";")
stats_curie <- read.csv("./../all_results_curie.csv", header=TRUE, sep=";")
stats_davinci <- read.csv("./../all_results_davinci.csv", header=TRUE, sep=";")

# Combine all results
stats <- rbind(stats_ada, stats_curie)
stats_inc_davinci <- rbind(stats_ada, stats_curie, stats_davinci)

######################### Adding derived value #############################################


## Add numeric correspondance for base model
stats$Base_Model_O <- stats$Base_Model
stats["Base_Model_O"][stats["Base_Model"] == "ada"] <- 0
stats["Base_Model_O"][stats["Base_Model"] == "curie"] <- 1
stats["Base_Model_O"][stats["Base_Model"] == "davinci"] <- 2
stats$Base_Model_O <- as.numeric(stats$Base_Model_O )
## Add numeric correspondance for base model
stats_inc_davinci$Base_Model_O <- stats_inc_davinci$Base_Model
stats_inc_davinci["Base_Model_O"][stats_inc_davinci["Base_Model"] == "ada"] <- 0
stats_inc_davinci["Base_Model_O"][stats_inc_davinci["Base_Model"] == "curie"] <- 1
stats_inc_davinci["Base_Model_O"][stats_inc_davinci["Base_Model"] == "davinci"] <- 2
stats_inc_davinci$Base_Model_O <- as.numeric(stats_inc_davinci$Base_Model_O )

plot_cor <- function(cor_mat){
  col<- colorRampPalette(c("blue", "white", "red"))(20)
  corrplot(cor_mat, addCoef.col = "black", # Add coefficient of correlation
           tl.col="black", tl.srt=45, #Text label color and rotation
           type="upper", order="hclust", diag=FALSE, col=col)
}

###########################################################################################
# The correlations for ada and curie
stats_temp <- stats %>% select(Base_Model_O, Epochs, Diffs, EOs, Pertubation, Number_Tokens,  Epochs, faulty_graphs, faulty_mm, token_count)
cors <- cor(stats_temp, method="pearson")
plot_cor(cors)

# Some graphs
pairs(stats_temp[, c("faulty_graphs", "Number_Tokens", "Pertubation", "Epochs")])

# The correlations for all three models
stats_temp_all <- stats_inc_davinci %>% filter(Pertubation==1.0) %>% select(Base_Model_O, Epochs, Diffs, EOs, Number_Tokens,  Epochs, faulty_graphs, faulty_mm, token_count)
cors <- cor(stats_temp_all, method="pearson")
plot_cor(cors)

# Check for significant dependency on the model
cor.test(stats_temp$faulty_graphs, stats_temp$Base_Model_O)

cor.test(stats_temp_all$faulty_graphs, stats_temp_all$Base_Model_O)

# Check for significant dependency on the Number of Tokens in the dataset
cor.test(stats_temp$faulty_graphs, stats_temp$Number_Tokens)

cor.test(stats_temp_all$faulty_graphs, stats_temp_all$Number_Tokens)


# Check for significant dependency on the Number of Epochs
cor.test(stats_temp$faulty_graphs, stats_temp$Epochs)

cor.test(stats_temp_all$faulty_graphs, stats_temp_all$Epochs)


# Overall percentage of all correct serializations

data_values = count(stats_inc_davinci)
mean(stats_inc_davinci$faulty_graphs)
min(stats_inc_davinci$faulty_graphs)
max(stats_inc_davinci$faulty_graphs)

mean(stats_inc_davinci$faulty_mm)
min(stats_inc_davinci$faulty_mm)
max(stats_inc_davinci$faulty_mm)

zero_faulty = count(stats_inc_davinci  %>% filter(faulty_graphs==0))
zero_faulty_all = count(stats_inc_davinci  %>% filter(faulty_graphs==0 & faulty_mm==0))

zero_faulty_relativ = zero_faulty/data_values
zero_faulty_all_relativ = zero_faulty_all/data_values

hist(stats_inc_davinci$faulty_graphs, breaks = 10)



corstarsl(stats_temp, type="spearman")
xtable(corstarsl(stats_temp)) #Latex code

corstarsl(stats_temp_all, type="spearman")
xtable(corstarsl(stats_temp_all)) #Latex code


# Problem cases i.e., more than 5 invalid graphs:
problem_sets <- stats %>% filter(faulty_graphs > 1 & EOs > 31)
count(problem_sets)


# For the generated graphs that do not conform to the meta-model, we observe low pearson but significant spearman correlation with the perturbation. We have to closer analyze this relationship.
plot.new()
# nb of eos vs. nb of components per diff
avg_faulty_mm <- aggregate(faulty_mm  ~ Pertubation, data=stats_temp, mean, na.rm=TRUE)
plot( stats_temp %>% select(Pertubation, faulty_mm ), col="red")
lines(avg_faulty_mm, col="black")

# From this graph, we clearly see, that it's just an outlier that fooled the pearson correlation



