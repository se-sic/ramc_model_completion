#!/bin/bash

# Dataset location
LOCATION='./case_studies/paper_fsm_hard/'

################## Step 1 ########################
# Read datasets and create connected components  #
# Input: Datasets + CSV describing the datasets  #
# Output: Connected Components per Dataset + CSV # 
#  CSV(updated by some parameters for the graphs)#
# TODO add information if the correct graph appears in database already#
##################################################
echo "Step 1: Read datasets and compute connected components."
python ./compute_connected_components.py $LOCATION'diffgraphs/' $LOCATION'results/components/' $LOCATION'correct_graphs/' False