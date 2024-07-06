#!/bin/bash

# Dataset location
LOCATION='./case_studies/paper_curie/'
DS_NAME=''
RESULTS_PATH=''

################## Step 1 ########################
# Generate pattern completion from the LM        #
# Input: Model + Test Dataset                    #
# Output: Evaluation of the completion           #
##################################################
echo "Step 1: Evaluate model completions."

python ./complete_model_diff.py $LOCATION/$DS_BAME $LOCATION/$RESULTS_PATH

