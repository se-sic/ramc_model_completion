#!/bin/bash

################ Pipeline Description ############
# This Pipeline generate input data (prompt) for #
# a few shot learning of changes using causal LLM#
#                                                #
# Input: Datasets + CSV describing the datasets  #
# Output: JSON-Lists with prompts for few-show   # 
#         experiments + some meta-data           #
##################################################

# Dataset location
#LOCATION_DIFFGRAPHS='./../model_completion_dataset/Siemens_Mobility/results/diffgraphs/'
#LOCATION_CONNECTED_COMPONETS='./../model_completion_dataset/Siemens_Mobility/results/components/'
#LOCATION_COMPLETION_DATASET='./../model_completion_dataset/Siemens_Mobility/results/fine_tune_ds/'
#LOCATION_SAMPLED_COMPLETION_DATASET='./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/'
#LOCATION_VECTOR_DB='./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/vector_db/'
#LOCATION_FEW_SHOT_DATASET='./../model_completion_dataset/Siemens_Mobility/results/few_shot_samples/'


#LOCATION_DIFFGRAPHS='./../model_completion_dataset/revision/diffgraphs/'
#LOCATION_CONNECTED_COMPONETS='./../model_completion_dataset/revision/components/'
#LOCATION_COMPLETION_DATASET='./../model_completion_dataset/revision/fine_tune_ds/'
#LOCATION_SAMPLED_COMPLETION_DATASET='./../model_completion_dataset/revision/experiment_samples/'
#LOCATION_VECTOR_DB='./../model_completion_dataset/revision/experiment_samples/vector_db/'
#LOCATION_FEW_SHOT_DATASET='./../model_completion_dataset/revision/few_shot_samples/'

LOCATION_DIFFGRAPHS='./../model_completion_dataset/Synthetic/diffgraphs/'
LOCATION_CONNECTED_COMPONETS='./../model_completion_dataset/Synthetic/components/'
LOCATION_COMPLETION_DATASET='./../model_completion_dataset/Synthetic/fine_tune_ds/'
LOCATION_SAMPLED_COMPLETION_DATASET='./../model_completion_dataset/Synthetic/experiment_samples/'
LOCATION_VECTOR_DB='./../model_completion_dataset/Synthetic/vector_db/'
LOCATION_FEW_SHOT_DATASET='./../model_completion_dataset/Synthetic/few_shot_samples/'

################## Step 1 ########################
# Read datasets and create connected components  #
# Input: Datasets + CSV describing the datasets  #
# Output: Connected Components per Dataset + CSV # 
#  CSV(updated by some parameters for the graphs)#
##################################################
echo "Step 1: Read datasets and compute connected components."
python ./compute_connected_components.py $LOCATION_DIFFGRAPHS $LOCATION_CONNECTED_COMPONETS

################## Step 2 ########################
# Serialize Graphs for the Fine-Tuning           #
# Input: CC + CSV describing the datasets        #
# Output: Serialized Datasets/training data +    # 
#  CSV(+serialization strategy parameters, n_tok)#
##################################################
echo "Step 2: Serialize Graphs for and generate prompt-completion pairs."
python ./create_finetune_training_sets.py $LOCATION_CONNECTED_COMPONETS $LOCATION_COMPLETION_DATASET

################## Step 3 ########################
# Pre-Sample: Select some samples and filter     #
# Input: JSONL prompt-completion pairs + meta    #
# Output: Sampled JSONL                        #
##################################################
echo "Step 3: Pre-Sample: Select some samples and filter."
python ./sample.py presample $LOCATION_COMPLETION_DATASET $LOCATION_SAMPLED_COMPLETION_DATASET

################## Step 4 ########################
# Create Vector DB for the sampled graphs        #
# Input: Sampled JSONL                           #
# Output: Vector DB                              #
##################################################
echo "Step 4: Create Vector DB for the sampled graphs."
python ./vector_database.py $LOCATION_SAMPLED_COMPLETION_DATASET $LOCATION_VECTOR_DB

################## Step 5 ########################
# Generate Few-Shot + Completion Sample prompts  #
# Input: Vector DB                               #
# Output: JSONL with prompts and meta-data       #
##################################################
echo "Step 5: Final sampling: Select some test samples and collect similar prompts."
python ./sample.py finalsample $LOCATION_VECTOR_DB $LOCATION_FEW_SHOT_DATASET