#!/bin/bash

# Dataset location
LOCATION='./case_studies/paper_curie/'

################## Step 1 ########################
# Read datasets and create connected components  #
# Input: Datasets + CSV describing the datasets  #
# Output: Connected Components per Dataset + CSV # 
#  CSV(updated by some parameters for the graphs)#
# TODO add information if the correct graph appears in database already#
##################################################
echo "Step 1: Read datasets and compute connected components."
python ./compute_connected_components.py $LOCATION'diffgraphs/' $LOCATION'results/components/' $LOCATION'correct_graphs/' False

################## Step 1b (Optional)##############
# Reduce tokens                                  #
# Input: CC                                      #
# Output: CCs with tokens replaced               #
##################################################
echo "Step 1b: Replacing graph labels (to save tokens)..."
echo "TODO: This is currently hardcoded for the Simple Component Model"
python ./transform_SCM.py $LOCATION'results/components/' enc


################## Step 2 ########################
# Serialize Graphs for the Fine-Tuning           #
# Input: CC + CSV describing the datasets        #
# Output: Serialized Datasets/training data +    # 
#  CSV(+serialization strategy parameters, n_tok)#
##################################################
echo "Step 2: Serialize Graphs for the Fine-Tuning"
python ./create_finetune_training_sets.py $LOCATION'results/components/' $LOCATION'results/finetune_ds/'

################## Step 3 ########################
# Trigger fine-tuning                            #
##################################################
#echo "Step 3: Execute Fine-Tuning jobs"
#python ./create_finetuning_jobs.py $LOCATION'results/finetune_ds/' $LOCATION'results/finetune/jobs/' 4
