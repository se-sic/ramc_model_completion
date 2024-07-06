#!/bin/bash

# Dataset location
LOCATION='./case_studies/paper_davinci/'

################## Step 1 ########################
# Generate pattern candidates from the LM        #
# Input: model                                   #
# Output: pattern candidates + meta-info         #
##################################################
echo "Step 1: Generate model transformation candidates and evalute"‚

python ./generate_model_transformation_candidates.py $LOCATION'results/all_results.csv' $LOCATION'correct_graphs_transformed/' $LOCATION

