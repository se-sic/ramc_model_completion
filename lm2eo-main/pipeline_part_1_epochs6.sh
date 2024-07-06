#!/bin/bash

# This script can only be run, if the other fine-tuning jobs (with less epochs) finished already.

# Dataset location
LOCATION='./case_studies/paper_davinci/'

################## Step 3b #######################
# Trigger fine-tuning for epochs > 4             #
##################################################
echo "Step 3: Execute Fine-Tuning jobs"
python ./create_finetuning_jobs.py $LOCATION'results/finetune_ds/' $LOCATION'results/finetune/jobs/' 6
