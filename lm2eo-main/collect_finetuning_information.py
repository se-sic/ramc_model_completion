#!/usr/bin/python  
import pandas as pd
import numpy as np
import sys, os

from common import load_results



def main(dataset_path):
    components_results_path = dataset_path + "/results/components/results.csv"
    finetune_dataset_results_path = dataset_path + "/results/finetune_ds/results.csv"
    jobs_file_path = dataset_path + "/results/finetune/jobs/results.csv"
    
    jobs_list = pd.read_csv(jobs_file_path, sep=";", header=0, keep_default_na=False)
    finetune_datasets = pd.read_csv(finetune_dataset_results_path, sep=";", header=0)
    components_info = pd.read_csv(components_results_path, sep=";", header=0)

    for idx, row in jobs_list.iterrows():
        finetunes_id = row['Finetune_Id']
        if finetunes_id == "NA":
            print(f"WARN: There are finetune jobs without id: DS: {row['Id']} - SS: {row['Strategy']} - Base Model: {row['Base_Model']} - Epochs: {row['Epochs']}")
            continue
        # Load all finetune info
        finetune_info = load_results(finetunes_id)
        #data_set_id = row['Id']
        #serialization_strategy = row['Strategy']
        nb_epochs = row['Epochs']
        base_model = row['Base_Model']
        
        # Consistency check (since we have some of the information also parsed)
        #assert str(nb_epochs) == finetune_info['nb_epochs']
        #assert base_model == finetune_info['base_model']
        
        jobs_list.loc[idx, 'Cost'] = finetune_info['cost']
        jobs_list.loc[idx, 'Finetune_Time'] = finetune_info['finetune_time']
        jobs_list.loc[idx, 'Average_Token_Acc'] = finetune_info['average_token_acc']
        jobs_list.loc[idx, 'model_name'] = finetune_info['model_name']
        
    # Now, merge the results with all other fine-tune dataset information
    # dataset_id + serialization_strategy are a key for the finetune_datasets info
    inner_merged_total = pd.merge(jobs_list, finetune_datasets, on=["Id","Strategy"])
    # And, also merge the information from the dataset properties
    inner_merged_total = pd.merge(inner_merged_total, components_info, on=["Id"]) 
        
    inner_merged_total.to_csv(dataset_path + '/results_ft.csv', sep=";", index = False)
                

if __name__ == "__main__":
  if len(sys.argv) == 2:
    main(sys.argv[1])
  else:
    print("Unexpected number of arguments. At least dataset path.")
